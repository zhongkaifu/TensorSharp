using System.Reflection;
using System.Runtime.InteropServices;
using System.Text.Json;
using TensorSharp.Cuda;
using TensorSharp.Models;
using TensorSharp.Runtime;

internal sealed unsafe class Native : IDisposable
{
    private static readonly Assembly Cuda = typeof(Q35CudaEngine).Assembly;
    internal static readonly Type Kernels = Cuda.GetType("TensorSharp.Cuda.Q35Kernels", true)!;
    internal static readonly Type Planes = typeof(Q35CudaEngine).GetNestedType("Planes", BindingFlags.NonPublic)!;
    internal static readonly Type Plan = typeof(Q35CudaEngine).GetNestedType("Q81RowPlan", BindingFlags.NonPublic)!;
    private static readonly Type Driver = Cuda.GetType("TensorSharp.Cuda.Interop.CudaDriverApi", true)!;
    private static readonly Type Module = Cuda.GetType("TensorSharp.Cuda.CudaModule", true)!;
    private delegate int Allocate(out IntPtr pointer, UIntPtr bytes);
    private delegate int Free(IntPtr pointer);
    private delegate int Copy(IntPtr destination, IntPtr source, UIntPtr bytes);
    internal delegate void Quant(IntPtr input, IntPtr quant, IntPtr scales, int width, int rows, IntPtr stream);
    internal delegate void Gemv(IntPtr codes, IntPtr scales, int outputWidth, int width, IntPtr quant, IntPtr activationScales, float alpha, IntPtr output, int rows, IntPtr stream);
    private static readonly Allocate Alloc = Oracle.Method(Driver, "cuMemAlloc", true, typeof(int), typeof(IntPtr).MakeByRefType(), typeof(UIntPtr)).CreateDelegate<Allocate>();
    private static readonly Free Release = Oracle.Method(Driver, "cuMemFree", true, typeof(int), typeof(IntPtr)).CreateDelegate<Free>();
    private static readonly Copy HtoD = Oracle.Method(Driver, "cuMemcpyHtoD", true, typeof(int), typeof(IntPtr), typeof(IntPtr), typeof(UIntPtr)).CreateDelegate<Copy>();
    private static readonly Copy DtoH = Oracle.Method(Driver, "cuMemcpyDtoH", true, typeof(int), typeof(IntPtr), typeof(IntPtr), typeof(UIntPtr)).CreateDelegate<Copy>();
    private static readonly MethodInfo Candidate = Oracle.Method(typeof(Q35CudaEngine), "ProjectQ81Ranges", false, typeof(void), typeof(string), Planes.MakeByRefType(), typeof(IntPtr), typeof(IntPtr), Plan);
    private static readonly MethodInfo MakePlan = Oracle.Method(typeof(Q35CudaEngine), "PlanVerifyDecodeQ81", true, Plan, typeof(int[]), typeof(bool[]));
    private readonly object referenceKernels;
    private readonly Q35CudaEngine engine;
    internal readonly Quant Quantize;
    internal readonly Gemv Multiply;
    internal readonly IntPtr Stream;
    internal delegate void Norm(IntPtr x, IntPtr delta, IntPtr weights, IntPtr output, int rows, int width, float eps, IntPtr stream);
    internal delegate void FusedNorm(IntPtr x, IntPtr delta, IntPtr weights, float eps, IntPtr codes, IntPtr scales, IntPtr rowScale, int width, int rows, IntPtr stream);
    internal readonly Norm Normalize;
    internal readonly FusedNorm NormalizeFp4;
    internal readonly object CandidateModule, ReferenceModule;
    internal static void CheckContract()
    {
        Oracle.Require(Candidate.GetParameters()[1].IsIn && Candidate.IsPrivate, "Candidate must be the actual private readonly-Planes method.");
        foreach (var (name, type) in new[] { ("Codes", typeof(IntPtr)), ("Scales", typeof(IntPtr)), ("Rows", typeof(int)), ("K", typeof(int)) }) _ = Oracle.Field(Planes, name, type, nonpublic: false);
        _ = Oracle.Method(Kernels, "QuantizeQ81", false, typeof(void), typeof(IntPtr), typeof(IntPtr), typeof(IntPtr), typeof(int), typeof(int), typeof(IntPtr));
        _ = Oracle.Method(Kernels, "AddRmsNorm", false, typeof(void), typeof(IntPtr), typeof(IntPtr), typeof(IntPtr), typeof(IntPtr), typeof(int), typeof(int), typeof(float), typeof(IntPtr));
        _ = Oracle.Method(Kernels, "AddRmsNormNvfp4", false, typeof(void), typeof(IntPtr), typeof(IntPtr), typeof(IntPtr), typeof(float), typeof(IntPtr), typeof(IntPtr), typeof(IntPtr), typeof(int), typeof(int), typeof(IntPtr));
        _ = Oracle.Method(Kernels, "GemvNvfp4", false, typeof(void), typeof(IntPtr), typeof(IntPtr), typeof(int), typeof(int), typeof(IntPtr), typeof(IntPtr), typeof(float), typeof(IntPtr), typeof(int), typeof(IntPtr));
        _ = Oracle.Field(typeof(Q35CudaEngine), "_k", Kernels); _ = Oracle.Field(Kernels, "module", Module); _ = Oracle.Field(Module, "module", typeof(IntPtr));
        _ = Oracle.Method(Module, "LoadFromFile", true, Module, typeof(string), typeof(bool));
        _ = Oracle.Method(Kernels, "FromModule", true, Kernels, Module);
        Oracle.Require(Module.GetProperty("ImageSha256")?.PropertyType == typeof(string), "Image identity contract changed.");
    }
    internal static string Locate()
    {
        var type = Cuda.GetType("TensorSharp.Cuda.CudaKernels", true)!;
        return (string?)Oracle.Method(type, "LocatePtxPath", true, typeof(string), typeof(string)).Invoke(null, new object[] { "tensorsharp_q35_kernels.ptx" }) ?? throw new IOException("Q35 PTX not found beside actual assembly.");
    }
    internal static object Provider()
    {
        string ptx = Locate(), fatbin = Path.ChangeExtension(ptx, ".fatbin"), sidecar = fatbin + ".sha256";
        Oracle.Require(Oracle.FileHash(fatbin) == "bbdaca53a0fb56ec78644356edfa981f6552e57c011f6b8cbfe015ac7b608515", "Actual located Q35 fatbin must be original BBD.");
        Oracle.Require(File.ReadAllText(sidecar).Trim().Equals(Oracle.FileHash(ptx), StringComparison.OrdinalIgnoreCase), "PTX/fatbin sidecar mismatch.");
        return new { ptx = new { path = ptx, bytes = new FileInfo(ptx).Length, sha256 = Oracle.FileHash(ptx) }, fatbin = new { path = fatbin, bytes = new FileInfo(fatbin).Length, sha256 = Oracle.FileHash(fatbin) }, sidecar = new { path = sidecar, sha256 = Oracle.FileHash(sidecar) }, loader_actual_image_sha_is_not_inferred = true };
    }
    private static object DescribeModule(object module, object kernels)
    {
        var get = Oracle.Method(Module, "GetFunction", false, typeof(IntPtr), typeof(string));
        var arch = Oracle.Method(Module, "CompiledArchOf", false, typeof(int), typeof(string));
        var entries = Enumerable.Range(1, 8).Select(n => $"ts_q35_gemv_nvfp4_{n}").Prepend("ts_q35_quantize_q81_f32").Concat(new[] { "ts_q35_add_rmsnorm_f32", "ts_q35_add_rmsnorm_nvfp4_f32", "ts_q35_gemv_q6k_1", "ts_q35_gemm_q6k_1", "ts_q35_gemm_q6k_2" }).ToDictionary(n => n, n => (object)new { handle = ((IntPtr)get.Invoke(module, new object[] { n })!).ToInt64(), reported_compiled_arch = (int)arch.Invoke(module, new object[] { n })! });
        var gemv = (IntPtr[])Oracle.Field(Kernels, "gemv", typeof(IntPtr[])).GetValue(kernels)!;
        Oracle.Require(Enumerable.Range(1, 8).All(n => gemv[n] == (IntPtr)get.Invoke(module, new object[] { $"ts_q35_gemv_nvfp4_{n}" })!), "Wrapper GEMV entries differ from identified module entries.");
        return new { module_handle = ((IntPtr)Oracle.Field(Module, "module", typeof(IntPtr)).GetValue(module)!).ToInt64(), image_sha256_property = (string?)Module.GetProperty("ImageSha256")!.GetValue(module), entries, rows_per_warp = (int[])Oracle.Field(Kernels, "gemvRows", typeof(int[])).GetValue(kernels)!, note = "Candidate null ImageSha256 is an explicit loader limitation, not a direct driver image hash." };
    }
    internal Native(Q35CudaEngine engine)
    {
        CheckContract(); this.engine = engine;
        var candidate = Oracle.Field(typeof(Q35CudaEngine), "_k", Kernels).GetValue(engine)!;
        CandidateModule = DescribeModule(Oracle.Field(Kernels, "module", Module).GetValue(candidate)!, candidate);
        object module = Oracle.Method(Module, "LoadFromFile", true, Module, typeof(string), typeof(bool)).Invoke(null, new object[] { Locate(), true })!;
        object? created = null;
        try
        {
        referenceKernels = created = Oracle.Method(Kernels, "FromModule", true, Kernels, Module).Invoke(null, new[] { module })!;
        ReferenceModule = DescribeModule(module, referenceKernels);
        Oracle.Require(((string?)Module.GetProperty("ImageSha256")!.GetValue(module))?.ToLowerInvariant() == "bbdaca53a0fb56ec78644356edfa981f6552e57c011f6b8cbfe015ac7b608515", "Identified independent reference image must actually be BBD (not PTX fallback).");
        Quantize = Oracle.Method(Kernels, "QuantizeQ81", false, typeof(void), typeof(IntPtr), typeof(IntPtr), typeof(IntPtr), typeof(int), typeof(int), typeof(IntPtr)).CreateDelegate<Quant>(referenceKernels);
        Multiply = Oracle.Method(Kernels, "GemvNvfp4", false, typeof(void), typeof(IntPtr), typeof(IntPtr), typeof(int), typeof(int), typeof(IntPtr), typeof(IntPtr), typeof(float), typeof(IntPtr), typeof(int), typeof(IntPtr)).CreateDelegate<Gemv>(referenceKernels);
        Normalize = Oracle.Method(Kernels, "AddRmsNorm", false, typeof(void), typeof(IntPtr), typeof(IntPtr), typeof(IntPtr), typeof(IntPtr), typeof(int), typeof(int), typeof(float), typeof(IntPtr)).CreateDelegate<Norm>(referenceKernels);
        NormalizeFp4 = Oracle.Method(Kernels, "AddRmsNormNvfp4", false, typeof(void), typeof(IntPtr), typeof(IntPtr), typeof(IntPtr), typeof(float), typeof(IntPtr), typeof(IntPtr), typeof(IntPtr), typeof(int), typeof(int), typeof(IntPtr)).CreateDelegate<FusedNorm>(referenceKernels);
        var stream = typeof(Q35CudaEngine).GetProperty("Stream", BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.DeclaredOnly);
        Oracle.Require(stream?.PropertyType == typeof(IntPtr), "Stream contract differs."); Stream = (IntPtr)stream!.GetValue(engine)!;
        }
        catch { if (created is IDisposable owned) owned.Dispose(); else ((IDisposable)module).Dispose(); throw; }
    }
    internal void UploadScratch(string field, byte[] data)
    {
        Oracle.Require(field is "_x" or "_xn" && data.Length <= (long)engine.MixedRowCapacity * engine.HiddenSize * 4, "Scratch upload bounds.");
        var pointer = (IntPtr)Oracle.Field(typeof(Q35CudaEngine), field, typeof(IntPtr)).GetValue(engine)!;
        fixed (byte* p = data) Oracle.Require(HtoD(pointer, (IntPtr)p, new UIntPtr((uint)data.Length)) == 0, "Scratch HtoD failed.");
    }
    internal byte[] Scratch(string field, int bytes)
    {
        Oracle.Require(field is "_x" or "_xn" && bytes <= (long)engine.MixedRowCapacity * engine.HiddenSize * 4, "Scratch readback bounds.");
        return Download((IntPtr)Oracle.Field(typeof(Q35CudaEngine), field, typeof(IntPtr)).GetValue(engine)!, bytes);
    }
    internal void Renorm(IntPtr weights, int start, int rows, int total)
    {
        Oracle.Method(typeof(Q35CudaEngine), "RenormalizeForcedRows", false, typeof(void), typeof(IntPtr), Plan).Invoke(engine, new object[] { weights, MakeRangePlan(start, rows, total) }); engine.Synchronize();
    }
    internal static T Plane<T>(object plane, string name) => (T)Oracle.Field(Planes, name, typeof(T), nonpublic: false).GetValue(plane)!;
    internal static object MakeRangePlan(int start, int rows, int total) => MakePlan.Invoke(null, new object[] { new[] { start, rows, checked(total - start - rows) }, new[] { false, true, false } })!;
    internal void Chunked(string role, object plane, IntPtr input, IntPtr output, int start, int rows, int total)
    {
        Oracle.Method(typeof(Q35CudaEngine), "ResetProjectionDispatchStats", false, typeof(void)).Invoke(engine, null);
        Candidate.Invoke(engine, new object[] { role, plane, input, output, MakeRangePlan(start, rows, total) });
        engine.Synchronize();
    }
    internal byte[] CandidateQuantTail(int k, int rows, bool scales)
    {
        string field = scales ? "_q8d" : "_q8"; int bytes = checked(rows * (scales ? k / 32 * 4 : k));
        var pointer = (IntPtr)Oracle.Field(typeof(Q35CudaEngine), field, typeof(IntPtr)).GetValue(engine)!; return Download(pointer, bytes);
    }
    internal static byte[] Download(IntPtr pointer, int bytes) { var result = new byte[bytes]; fixed (byte* p = result) Oracle.Require(DtoH((IntPtr)p, pointer, new UIntPtr((uint)bytes)) == 0, "DtoH failed."); return result; }
    public void Dispose() { try { engine.Synchronize(); } finally { ((IDisposable)referenceKernels).Dispose(); } }
    internal sealed class GpuBuffer : IDisposable
    {
        internal const int Guard = 256;
        internal readonly int Bytes; internal readonly byte Poison; private readonly Action sync; internal IntPtr Base; internal IntPtr Data => Base + Guard;
        internal GpuBuffer(int bytes, byte poison, Action synchronize)
        {
            Oracle.Require(bytes > 0, "GPU buffer must be nonempty."); Bytes = bytes; Poison = poison; sync = synchronize;
            Oracle.Require(Alloc(out Base, new UIntPtr(checked((uint)(bytes + 2 * Guard)))) == 0, "Allocation failed.");
            try { var fill = Enumerable.Repeat(poison, checked(bytes + 2 * Guard)).ToArray(); UploadAll(fill); }
            catch { if (Base != IntPtr.Zero) Release(Base); Base = IntPtr.Zero; throw; }
        }
        internal void UploadAll(byte[] bytes) { Oracle.Require(bytes.Length == Bytes + 2 * Guard, "Upload bounds."); fixed (byte* p = bytes) Oracle.Require(HtoD(Base, (IntPtr)p, new UIntPtr((uint)bytes.Length)) == 0, "HtoD failed."); }
        internal void UploadPayload(byte[] bytes) { Oracle.Require(bytes.Length == Bytes, "Payload size."); fixed (byte* p = bytes) Oracle.Require(HtoD(Data, (IntPtr)p, new UIntPtr((uint)bytes.Length)) == 0, "HtoD failed."); }
        internal byte[] Read() => Download(Base, checked(Bytes + 2 * Guard));
        internal bool Guards(byte[] bytes, int activeOffset, int activeBytes) => Guarded(bytes, Bytes, Poison, activeOffset, activeBytes);
        internal static bool Guarded(byte[] bytes, int payload, byte poison, int activeOffset, int activeBytes)
        {
            if (bytes.Length != payload + 2 * Guard || activeOffset < 0 || activeBytes < 0 || (long)activeOffset + activeBytes > payload) return false;
            int start = Guard + activeOffset, end = start + activeBytes;
            return bytes.Take(start).All(x => x == poison) && bytes.Skip(end).All(x => x == poison);
        }
        public void Dispose() { if (Base != IntPtr.Zero) { try { sync(); } finally { IntPtr owned = Base; Base = IntPtr.Zero; Oracle.Require(Release(owned) == 0, "Free failed."); } } }
    }
}

internal static class Component
{
    internal static float[] Values(int rows, int k, int pattern)
    {
        var x = new float[checked(rows * k)];
        for (int i = 0; i < x.Length; i++)
        {
            uint hash = unchecked((uint)(i + 1) * 2654435761u + (uint)pattern * 2246822519u);
            x[i] = ((hash >> 8) % 2049 - 1024.0f) / 1024 * (1 << ((i / k) % 4));
            if (pattern == 1 && i % 11 == 0) x[i] = BitConverter.UInt32BitsToSingle((i % 4) switch { 0 => 0u, 1 => 0x80000000u, 2 => 1u, _ => 0x80000001u });
            if (pattern == 2 && i % 29 == 0) x[i] = BitConverter.UInt32BitsToSingle((i % 3) switch { 0 => 0x7fc00001u, 1 => 0x7f800000u, _ => 0xff800000u });
        }
        return x;
    }
    internal static object Run(Q35CudaEngine engine, Native api, string output)
    {
        Directory.CreateDirectory(output); var records = new List<object>(); bool pass = true;
        var layers = (Array)Oracle.Field(typeof(Q35CudaEngine), "_layers").GetValue(engine)!;
        int gdn = Enumerable.Range(0, layers.Length).First(i => (bool)Oracle.Field(layers.GetValue(i)!.GetType(), "Recurrent", typeof(bool), nonpublic: false).GetValue(layers.GetValue(i))!);
        int attn = Enumerable.Range(0, layers.Length).First(i => !(bool)Oracle.Field(layers.GetValue(i)!.GetType(), "Recurrent", typeof(bool), nonpublic: false).GetValue(layers.GetValue(i))!);
        foreach (var (layerIndex, role, label) in new[] { (gdn, "InProj", "gdn-in"), (attn, "InProj", "attn-in"), (gdn, "OutProj", "gdn-out"), (attn, "OutProj", "attn-out"), (gdn, "GateUp", "gate-up"), (gdn, "Down", "down") })
        {
            object layer = layers.GetValue(layerIndex)!, w = Oracle.Field(layer.GetType(), role, Native.Planes, nonpublic: false).GetValue(layer)!;
            int k = Native.Plane<int>(w, "K"), outDim = Native.Plane<int>(w, "Rows"); var codes = Native.Plane<IntPtr>(w, "Codes"); var weightScale = Native.Plane<IntPtr>(w, "Scales");
            const int first = 3;
            foreach (int selected in new[] { 8, 9, 10, 11, 12, 13, 14, 15, 17 })
            for (int pattern = 0; pattern < (selected == 17 ? 3 : 1); pattern++)
            {
                int total = first + selected + 3;
                string id = label + "-n" + selected + "-p" + pattern; int xBytes = checked(total * k * 4), qBytes = total * k, sBytes = total * (k / 32) * 4, yBytes = checked(total * outDim * 4);
                using var input = new Native.GpuBuffer(xBytes, 0x17, engine.Synchronize); var values = Values(total, k, pattern); var inputBytes = MemoryMarshal.AsBytes(values.AsSpan()).ToArray(); input.UploadPayload(inputBytes);
                using var q8 = new Native.GpuBuffer(qBytes, 0x91, engine.Synchronize); using var q1 = new Native.GpuBuffer(qBytes, 0x62, engine.Synchronize); using var s8 = new Native.GpuBuffer(sBytes, 0xa3, engine.Synchronize); using var s1 = new Native.GpuBuffer(sBytes, 0x54, engine.Synchronize);
                using var candidate = new Native.GpuBuffer(yBytes, 0xc5, engine.Synchronize); using var r8 = new Native.GpuBuffer(yBytes, 0x36, engine.Synchronize); using var r1 = new Native.GpuBuffer(yBytes, 0xe7, engine.Synchronize);
                for (int done = 0; done < selected; done += 8) { int n = Math.Min(8, selected - done), row = first + done; api.Quantize(input.Data + row * k * 4, q8.Data + row * k, s8.Data + row * (k / 32) * 4, k, n, api.Stream); }
                for (int row = first; row < first + selected; row++) api.Quantize(input.Data + row * k * 4, q1.Data + row * k, s1.Data + row * (k / 32) * 4, k, 1, api.Stream);
                engine.Synchronize(); api.Chunked(role, w, input.Data, candidate.Data, first, selected, total); var actualDispatch = engine.ProjectionDispatchStats;
                for (int done = 0; done < selected; done += 8) { int n = Math.Min(8, selected - done), row = first + done; api.Multiply(codes, weightScale, outDim, k, q8.Data + row * k, s8.Data + row * (k / 32) * 4, 1, r8.Data + row * outDim * 4, n, api.Stream); }
                for (int row = first; row < first + selected; row++) api.Multiply(codes, weightScale, outDim, k, q1.Data + row * k, s1.Data + row * (k / 32) * 4, 1, r1.Data + row * outDim * 4, 1, api.Stream);
                engine.Synchronize(); var qb8 = q8.Read(); var qb1 = q1.Read(); var sb8 = s8.Read(); var sb1 = s1.Read(); var cy = candidate.Read(); var y8 = r8.Read(); var y1 = r1.Read();
                byte[] Active(byte[] bytes, int stride) => bytes.AsSpan(Native.GpuBuffer.Guard + first * stride, selected * stride).ToArray();
                float[] ToFloat(byte[] bytes) { var v = new float[bytes.Length / 4]; Buffer.BlockCopy(bytes, 0, v, 0, bytes.Length); return v; }
                var a = ToFloat(Active(cy, outDim * 4)); var b = ToFloat(Active(y8, outDim * 4)); var c = ToFloat(Active(y1, outDim * 4));
                bool quantEqual = Active(qb8, k).SequenceEqual(Active(qb1, k)) && Active(sb8, k / 32 * 4).SequenceEqual(Active(sb1, k / 32 * 4));
                int tailRows = selected % 8 == 0 ? 8 : selected % 8, tailStart = selected - tailRows;
                bool tail = api.CandidateQuantTail(k, tailRows, false).SequenceEqual(Active(qb8, k).AsSpan(tailStart * k, tailRows * k).ToArray()) && api.CandidateQuantTail(k, tailRows, true).SequenceEqual(Active(sb8, k / 32 * 4).AsSpan(tailStart * k / 32 * 4, tailRows * k / 32 * 4).ToArray());
                bool guards = q8.Guards(qb8, first * k, selected * k) && q1.Guards(qb1, first * k, selected * k) && s8.Guards(sb8, first * k / 32 * 4, selected * k / 32 * 4) && s1.Guards(sb1, first * k / 32 * 4, selected * k / 32 * 4)
                    && candidate.Guards(cy, first * outDim * 4, selected * outDim * 4) && r8.Guards(y8, first * outDim * 4, selected * outDim * 4) && r1.Guards(y1, first * outDim * 4, selected * outDim * 4);
                var inputRead = input.Read(); guards &= inputRead.AsSpan(Native.GpuBuffer.Guard, xBytes).SequenceEqual(inputBytes) && input.Guards(inputRead, 0, xBytes);
                var candidateVs8 = Oracle.Compare(b, a, selected, outDim); var n1Vs8 = Oracle.Compare(b, c, selected, outDim);
                var expectedCalls = Enumerable.Range(0, (selected + 7) / 8).Select(i => (first + 8 * i, Math.Min(8, selected - 8 * i))).ToArray();
                bool actualCalls = actualDispatch.Length == expectedCalls.Length && actualDispatch.All(d => d.Role == role && d.DType == "NVFP4_W4A8_Q8_1" && d.Calls == 1)
                    && actualDispatch.Select(d => (d.FirstRow, d.Rows)).Order().SequenceEqual(expectedCalls);
                bool ok = quantEqual && tail && guards && actualCalls && Oracle.Strict(candidateVs8) && Oracle.Strict(n1Vs8); pass &= ok;
                var rawFiles = new List<object>();
                foreach (var (name, bytes) in new[] { ("input", inputRead), ("quant8", qb8), ("quant1", qb1), ("scale8", sb8), ("scale1", sb1), ("candidate", cy), ("reference8", y8), ("reference1", y1) })
                { string file = id + "." + name + ".bin"; File.WriteAllBytes(Path.Combine(output, file), bytes); Oracle.Require(Oracle.FileHash(Path.Combine(output, file)) == Oracle.Hash(bytes), "Component raw write/readback mismatch."); rawFiles.Add(new { file, bytes = bytes.Length, sha256 = Oracle.Hash(bytes) }); }
                records.Add(new { id, layerIndex, role, k, outDim, total_rows = total, first_row = first, selected_rows = selected, pattern, strict_pass = ok, quant_bytes_and_scales_N1_N8 = quantEqual, candidate_final_tail_quant_matches = tail, guards, actual_host_dispatch_counts = actualDispatch, actualCalls, candidate_vs_explicit8tail = candidateVs8, explicit_N1_vs8tail = n1Vs8, reference_does_not_invoke_candidate_wrapper = true, raw_files = rawFiles });
                Oracle.Write(Path.Combine(output, "result.json"), new { strict_pass = pass, records, complete = false });
            }
        }
        var normalization = Norms(engine, api, layers.GetValue(gdn)!, output);
        pass &= Oracle.Strict(normalization);
        var result = new { strict_pass = pass, complete = true, fixtures = records.Count, records, normalization, tolerances = "NONE; any F32 word mismatch is FAIL even when top1 agrees", host_dispatch_is_not_profiler = true }; Oracle.Write(Path.Combine(output, "result.json"), result); return result;
    }
    private static object Norms(Q35CudaEngine engine, Native api, object layer, string output)
    {
        var desc = (Q35CudaEngine.ModelDesc)Oracle.Field(typeof(Q35CudaEngine), "_m", typeof(Q35CudaEngine.ModelDesc)).GetValue(engine)!;
        int e = engine.HiddenSize; const int rows = 23, first = 3, active = 17; int bytes = rows * e * 4; bool pass = true; var records = new List<object>();
        foreach (string name in new[] { "AttnNorm", "PostNorm" }) for (int pattern = 0; pattern < 2; pattern++)
        {
            var weights = (IntPtr)Oracle.Field(layer.GetType(), name, typeof(IntPtr), nonpublic: false).GetValue(layer)!;
            var input = MemoryMarshal.AsBytes(Values(rows, e, pattern).AsSpan()).ToArray(); var delta = MemoryMarshal.AsBytes(Values(rows, e, 3).Select(x => x * .125f).ToArray().AsSpan()).ToArray();
            using var a = new Native.GpuBuffer(bytes, 0x31, engine.Synchronize); using var b = new Native.GpuBuffer(bytes, 0x42, engine.Synchronize); using var d = new Native.GpuBuffer(bytes, 0x53, engine.Synchronize); using var y = new Native.GpuBuffer(bytes, 0x64, engine.Synchronize);
            using var codes = new Native.GpuBuffer(rows * e / 2, 0x75, engine.Synchronize); using var scales = new Native.GpuBuffer(rows * e / 16, 0x86, engine.Synchronize); using var rowScale = new Native.GpuBuffer(rows * 4, 0x97, engine.Synchronize);
            a.UploadPayload(input); b.UploadPayload(input); d.UploadPayload(delta);
            api.Normalize(a.Data, d.Data, weights, y.Data, rows, e, desc.Eps, api.Stream);
            api.NormalizeFp4(b.Data, d.Data, weights, desc.Eps, codes.Data, scales.Data, rowScale.Data, e, rows, api.Stream); engine.Synchronize();
            var ax = a.Read(); var bx = b.Read(); var reference = y.Read(); var actualResidual = bx.AsSpan(Native.GpuBuffer.Guard, bytes).ToArray();
            api.UploadScratch("_x", actualResidual); api.UploadScratch("_xn", Enumerable.Repeat((byte)0xa5, bytes).ToArray()); api.Renorm(weights, first, active, rows);
            var actual = api.Scratch("_xn", bytes); var xAfter = api.Scratch("_x", bytes);
            float[] F(byte[] q, int at, int n) { var result = new float[n / 4]; Buffer.BlockCopy(q, at, result, 0, n); return result; }
            var comparison = Oracle.Compare(F(reference, Native.GpuBuffer.Guard + first * e * 4, active * e * 4), F(actual, first * e * 4, active * e * 4), active, e);
            bool residual = ax.AsSpan(Native.GpuBuffer.Guard, bytes).SequenceEqual(actualResidual) && actualResidual.SequenceEqual(xAfter);
            var deltaAfter = d.Read();
            bool guards = a.Guards(ax, 0, bytes) && b.Guards(bx, 0, bytes) && d.Guards(deltaAfter, 0, bytes) && deltaAfter.AsSpan(Native.GpuBuffer.Guard, bytes).SequenceEqual(delta) && y.Guards(reference, 0, bytes)
                && actual.Take(first * e * 4).All(x => x == 0xa5) && actual.Skip((first + active) * e * 4).All(x => x == 0xa5)
                && codes.Guards(codes.Read(), 0, codes.Bytes) && scales.Guards(scales.Read(), 0, scales.Bytes) && rowScale.Guards(rowScale.Read(), 0, rowScale.Bytes);
            bool ok = residual && guards && Oracle.Strict(comparison); pass &= ok; var files = new List<object>();
            foreach (var (suffix, data) in new[] { ("standard-residual", ax), ("fused-residual", bx), ("standard-norm", reference), ("actual-private-renorm", actual), ("residual-after-private", xAfter), ("input", input), ("delta", delta) })
            { string file = $"norm-{name}-p{pattern}.{suffix}.bin"; File.WriteAllBytes(Path.Combine(output, file), data); Oracle.Require(Oracle.FileHash(Path.Combine(output, file)) == Oracle.Hash(data), "Norm raw write/readback mismatch."); files.Add(new { file, bytes = data.Length, sha256 = Oracle.Hash(data) }); }
            records.Add(new { name, pattern, rows, first, active, strict_pass = ok, residual_once_exact = residual, guards, comparison, raw_files = files });
            Oracle.Write(Path.Combine(output, "norms.json"), new { strict_pass = pass, complete = false, records });
        }
        var result = new { strict_pass = pass, complete = true, records }; Oracle.Write(Path.Combine(output, "norms.json"), result); return result;
    }

}

internal static class PartitionHostGuards
{
    internal const int Words = 16;
    internal const uint Poison = 0x7fa55a17;
    internal sealed record State(bool Logits, bool Drawn, bool Features, uint[] LogitsTail, uint[] DrawnTail, uint[] FeaturesTail)
    {
        public bool All => Logits && Drawn && Features;
    }
    internal static float[] PaintFloat(int active)
    {
        Oracle.Require(active >= 0, "Nonnegative host payload required.");
        var bits = Enumerable.Repeat(Poison, checked(active + Words)).ToArray();
        var values = new float[bits.Length];
        Buffer.BlockCopy(bits, 0, values, 0, checked(bits.Length * sizeof(uint)));
        return values;
    }
    internal static int[] PaintInt(int active)
    {
        Oracle.Require(active >= 0, "Nonnegative host payload required.");
        var bits = Enumerable.Repeat(Poison, checked(active + Words)).ToArray();
        var values = new int[bits.Length];
        Buffer.BlockCopy(bits, 0, values, 0, checked(bits.Length * sizeof(uint)));
        return values;
    }
    internal static State Inspect(float[] logits, int n, int[] drawn, int rows, float[] features, int f)
    {
        Oracle.Require(n >= 0 && rows >= 0 && f >= 0 && logits.Length == (long)n + Words && drawn.Length == (long)rows + Words && features.Length == (long)f + Words, "Exact host guard dimensions required.");
        // Inspect integer bits, never materialize signaling-NaN guard words as float values.
        var l = MemoryMarshal.Cast<float, uint>(logits.AsSpan(n, Words)).ToArray();
        var d = MemoryMarshal.Cast<int, uint>(drawn.AsSpan(rows, Words)).ToArray();
        var t = MemoryMarshal.Cast<float, uint>(features.AsSpan(f, Words)).ToArray();
        return new(l.All(x => x == Poison), d.All(x => x == Poison), t.All(x => x == Poison), l, d, t);
    }
    internal static int SelfTest()
    {
        int assertions = 0; var observed = new List<object>();
        void Check(bool ok) { assertions++; Oracle.Require(ok, "Host guard CPU assertion " + assertions); }
        foreach (int rows in new[] { 2, 8, 16 })
        {
            int n = rows * 248320, f = rows * 15360;
            var logits = PaintFloat(n); var drawn = PaintInt(rows); var features = PaintFloat(f);
            var before = Inspect(logits, n, drawn, rows, features, f); Check(before.All);
            MemoryMarshal.Cast<float, uint>(logits.AsSpan(0, n)).Fill(0x3f800001);
            drawn.AsSpan(0, rows).Fill(42);
            MemoryMarshal.Cast<float, uint>(features.AsSpan(0, f)).Fill(0x40000001);
            Check(Inspect(logits, n, drawn, rows, features, f).All);
            MemoryMarshal.Cast<float, uint>(logits.AsSpan(n, Words))[0] ^= 1;
            var badLogit = Inspect(logits, n, drawn, rows, features, f); Check(!badLogit.Logits && badLogit.Drawn && badLogit.Features);
            MemoryMarshal.Cast<float, uint>(logits.AsSpan(n, Words))[0] ^= 1;
            drawn[rows] ^= 1;
            var badDrawn = Inspect(logits, n, drawn, rows, features, f); Check(badDrawn.Logits && !badDrawn.Drawn && badDrawn.Features);
            drawn[rows] ^= 1;
            MemoryMarshal.Cast<float, uint>(features.AsSpan(f, Words))[0] ^= 1;
            var badFeature = Inspect(logits, n, drawn, rows, features, f); Check(badFeature.Logits && badFeature.Drawn && !badFeature.Features);
            MemoryMarshal.Cast<float, uint>(features.AsSpan(f, Words))[0] ^= 1;
            var restored = Inspect(logits, n, drawn, rows, features, f); Check(restored.All);
            observed.Add(new { rows, before, after_active_writes = restored, all_three_independent_corruptions_rejected = true });
        }
        Console.WriteLine(JsonSerializer.Serialize(new { diagnostic = "host-guard-bitpaint-v2", assertions, CPU_only = true, GPU_calls = 0, runtime = RuntimeInformation.FrameworkDescription, architecture = RuntimeInformation.ProcessArchitecture.ToString(), poison_u32 = Poison, guard_words = Words, observed }, Oracle.Json));
        return 0;
    }
}

internal static class Partition
{
    private sealed record Snapshot(float[] Logits, int[] Drawn, float[] Features, float[] HeadInput, byte[] HeadQuant, byte[] HeadScales);
    internal static object Run(Q35CudaEngine engine, Native api, string modelPath, string output)
    {
        Directory.CreateDirectory(output); var records = new List<object>(); var comparisons = new List<object>(); bool pass = true;
        var desc = (Q35CudaEngine.ModelDesc)Oracle.Field(typeof(Q35CudaEngine), "_m", typeof(Q35CudaEngine.ModelDesc)).GetValue(engine)!;
        engine.SetFeatureTaps(new[] { 0, desc.NLayer / 2, desc.NLayer - 1 }.Distinct().ToArray());
        using var gguf = new GgufFile(modelPath); var tokenizer = ModelBase.CreateTokenizerFromGguf(gguf);
        var text = tokenizer.Encode("A library contains books and each book has an author. Add two three and five to obtain ten. Tokyo is the capital of Japan. Every row follows the same input and position.\n", false).ToArray();
        Oracle.Require(text.Length >= 32, "Insufficient teacher tokens.");
        int[] Tokens(int start, int count) => Enumerable.Range(0, count).Select(i => text[(start + i) % text.Length]).ToArray();
        var prefix = Tokens(0, 7); var window = Tokens(13, 8).Concat(Tokens(31, 8)).ToArray();
        var teacher = Enumerable.Range(0, 8).Select(i => Tokens(47 + i * 2, 2)).ToArray();
        Oracle.Write(Path.Combine(output, "inputs.json"), new { prefix, window, teacher, kept = new[] { 8, 8 }, no_drafter = true, equal_prefix_lengths = true });
        var primary = engine.Active; var secondary = engine.CreateSlot(0); var slots = new[] { primary, secondary };
        int vocab = engine.VocabSize, featureWidth = engine.FeatureWidth;
        void State7() { engine.ResetSlot(primary); engine.ResetSlot(secondary); engine.Active = primary; engine.Forward(prefix, null); engine.CopySlot(primary, secondary); Oracle.Require(slots.All(s => s.NPast == 7), "Prefix state length drift."); }
        void SaveResult() => Oracle.Write(Path.Combine(output, "result.json"), new { strict_pass = pass, complete = false, records, comparisons, no_float_tolerance = true });
        Snapshot Capture(string id, Q35CudaEngine.Slot[] selected, int width, int[] tokens, bool verify)
        {
            int rows = selected.Length * width, n = checked(rows * vocab), f = checked(rows * featureWidth);
            var logits = PartitionHostGuards.PaintFloat(n); var features = PartitionHostGuards.PaintFloat(f); var drawn = PartitionHostGuards.PaintInt(rows);
            var hostGuardsBefore = PartitionHostGuards.Inspect(logits, n, drawn, rows, features, f);
            string guardBeforeFile = id + ".host-guards-before.json";
            Oracle.Write(Path.Combine(output, guardBeforeFile), hostGuardsBefore);
            Oracle.Require(hostGuardsBefore.Logits, "Logits host tail invalid before model call.");
            Oracle.Require(hostGuardsBefore.Drawn, "Drawn host tail invalid before model call.");
            Oracle.Require(hostGuardsBefore.Features, "Feature host tail invalid before model call.");
            int[] before = slots.Select(s => s.NPast).ToArray();
            if (verify) engine.Verify(selected, width, tokens, drawn, logits); else engine.DecodeBatch(selected, tokens, logits, drawn);
            if (featureWidth > 0 && verify) engine.ReadFeatureRows(features, rows);
            var hostGuardsAfter = PartitionHostGuards.Inspect(logits, n, drawn, rows, features, f);
            string guardAfterFile = id + ".host-guards-after.json";
            Oracle.Write(Path.Combine(output, guardAfterFile), hostGuardsAfter);
            bool guards = hostGuardsAfter.All;
            var data = logits[..n]; var ids = drawn[..rows]; var taps = verify ? features[..f] : Array.Empty<float>();
            var headBytes = api.Scratch("_xn", rows * engine.HiddenSize * 4); var headInput = new float[headBytes.Length / 4]; Buffer.BlockCopy(headBytes, 0, headInput, 0, headBytes.Length);
            var headQuant = api.CandidateQuantTail(engine.HiddenSize, rows, false); var headScales = api.CandidateQuantTail(engine.HiddenSize, rows, true);
            var rawFiles = new List<object>();
            foreach (string file in new[] { guardBeforeFile, guardAfterFile })
                rawFiles.Add(new { file, bytes = new FileInfo(Path.Combine(output, file)).Length, sha256 = Oracle.FileHash(Path.Combine(output, file)) });
            foreach (var (name, bytes) in new[] { ("f32", MemoryMarshal.AsBytes(data.AsSpan()).ToArray()), ("i32", MemoryMarshal.AsBytes(ids.AsSpan()).ToArray()), ("features-f32", MemoryMarshal.AsBytes(taps.AsSpan()).ToArray()), ("head-input-f32", headBytes), ("head-quant-u8", headQuant), ("head-scale-f32", headScales) })
            { string file = id + "." + name; File.WriteAllBytes(Path.Combine(output, file), bytes); Oracle.Require(Oracle.FileHash(Path.Combine(output, file)) == Oracle.Hash(bytes), "Raw snapshot write/readback hash mismatch."); rawFiles.Add(new { file, bytes = bytes.Length, sha256 = Oracle.Hash(bytes) }); }
            bool finite = data.All(float.IsFinite) && taps.All(float.IsFinite) && headInput.All(float.IsFinite), argmax = Enumerable.Range(0, rows).All(r => ids[r] == Oracle.Argmax(data, r * vocab, vocab));
            bool positions = verify ? before.SequenceEqual(slots.Select(s => s.NPast)) : before.Zip(slots, (p, s) => s.NPast == p + 1).All(x => x);
            pass &= guards && finite && argmax && positions;
            records.Add(new { id, verify, rows, vocab, featureWidth, guards, host_guard_diagnostic_version = 2, host_guard_before_all = hostGuardsBefore.All, logits_host_guard = hostGuardsAfter.Logits, drawn_host_guard = hostGuardsAfter.Drawn, features_host_guard = hostGuardsAfter.Features, finite, argmax, positions, positions_before = before, positions_after = slots.Select(s => s.NPast).ToArray(), raw_files = rawFiles,
                teacher_reference_and_candidate_both_use_DecodeBatch2 = !verify, head_rows = rows, head_mode = rows == 1 ? "Q6_K_DP4A" : rows <= 8 ? "Q6_K_INT8_MMA_8bucket" : "Q6_K_INT8_MMA_16bucket", feature_readback_performed = verify, actual_host_projection_dispatch = engine.ProjectionDispatchStats, host_API_not_kernel_profiler = true }); SaveResult();
            return new(data, ids, taps, headInput, headQuant, headScales);
        }
        void Check(string label, Snapshot a, Snapshot b, int rows)
        {
            Oracle.Require(a.Features.Length == b.Features.Length, "Feature snapshots are incomplete or of different kinds.");
            var logits = Oracle.Compare(a.Logits, b.Logits, rows, vocab); var features = featureWidth == 0 || a.Features.Length == 0 ? new { strict_pass = true } : (object)Oracle.Compare(a.Features, b.Features, rows, featureWidth);
            var headInput = Oracle.Compare(a.HeadInput, b.HeadInput, rows, engine.HiddenSize);
            bool headQuant = a.HeadQuant.SequenceEqual(b.HeadQuant), headScales = a.HeadScales.SequenceEqual(b.HeadScales), drawn = a.Drawn.SequenceEqual(b.Drawn);
            pass &= Oracle.Strict(logits) && Oracle.Strict(features) && Oracle.Strict(headInput) && headQuant && headScales && drawn;
            comparisons.Add(new { label, logits, features, drawn_strict = drawn, head_input_comparison = headInput, head_quant_bytes_equal = headQuant, head_scales_equal = headScales }); SaveResult();
        }
        Snapshot Join(Snapshot a, Snapshot b) => new(a.Logits.Concat(b.Logits).ToArray(), a.Drawn.Concat(b.Drawn).ToArray(), a.Features.Concat(b.Features).ToArray(), a.HeadInput.Concat(b.HeadInput).ToArray(), a.HeadQuant.Concat(b.HeadQuant).ToArray(), a.HeadScales.Concat(b.HeadScales).ToArray());
        try
        {
            State7();
            var a0 = Capture("abba-A0-N2", slots, 8, window, true);
            var b00 = Capture("abba-B0-slot0", new[] { primary }, 8, window[..8], true); var b01 = Capture("abba-B0-slot1", new[] { secondary }, 8, window[8..], true);
            var b10 = Capture("abba-B1-slot0", new[] { primary }, 8, window[..8], true); var b11 = Capture("abba-B1-slot1", new[] { secondary }, 8, window[8..], true);
            var a1 = Capture("abba-A1-N2", slots, 8, window, true);
            Check("N2-AA", a0, a1, 16); Check("N1-slot0-AA", b00, b10, 8); Check("N1-slot1-AA", b01, b11, 8); Check("N2-vs-separate-N1-beforeCommit", a0, Join(b00, b01), 16);
            // Required even though A1 was N2: make the final replay-buffer layout
            // explicit immediately before the N2 Commit. Never commit from lastN1.
            var precommit = Capture("N2-matching-reverify-beforeCommit", slots, 8, window, true); Check("N2-precommit-AA", a0, precommit, 16);
            engine.Commit(slots, 8, new[] { 8, 8 }); Oracle.Require(slots.All(s => s.NPast == 15), "N2 Commit position drift.");
            var continuation = new List<Snapshot>(); for (int i = 0; i < 8; i++) continuation.Add(Capture($"jointCommit-teacher-{i}", slots, 1, teacher[i], false));
            State7();
            Capture("separateCommit-slot0-verify", new[] { primary }, 8, window[..8], true); engine.Commit(new[] { primary }, 8, new[] { 8 });
            Capture("separateCommit-slot1-verify", new[] { secondary }, 8, window[8..], true); engine.Commit(new[] { secondary }, 8, new[] { 8 }); Oracle.Require(slots.All(s => s.NPast == 15), "N1 Commit position drift.");
            for (int i = 0; i < 8; i++) Check($"teacher-joint-vs-separateCommit-{i}", continuation[i], Capture($"separateCommit-teacher-{i}", slots, 1, teacher[i], false), 2);
        }
        finally { engine.Active = primary; engine.FreeSlot(secondary); }
        var result = new { strict_pass = pass, complete = true, records, comparisons, same_physical_slots = true, verify_prefix7_equal_maxKeys = true,
            last_N1_does_not_feed_N2_Commit = true, attention_reduction_limit = "Any mismatch remains strict FAIL. Equal prefixes hold maxKeys fixed; no ragged-K tolerance is substituted." };
        Oracle.Write(Path.Combine(output, "result.json"), result); return result;
    }
}

internal static class SelfTest
{
    internal static int Run()
    {
        int count = 0;
        void Check(bool ok) { count++; Oracle.Require(ok, "CPU self-test " + count); }
        Native.CheckContract(); count++;
        var clean = Enumerable.Repeat((byte)0xa5, 1024).ToArray(); Check(Native.GpuBuffer.Guarded(clean, 512, 0xa5, 64, 128));
        var active = (byte[])clean.Clone(); Array.Fill(active, (byte)0x5a, 256 + 64, 128); Check(Native.GpuBuffer.Guarded(active, 512, 0xa5, 64, 128));
        var before = (byte[])active.Clone(); before[255] ^= 1; Check(!Native.GpuBuffer.Guarded(before, 512, 0xa5, 64, 128));
        var after = (byte[])active.Clone(); after[256 + 64 + 128] ^= 1; Check(!Native.GpuBuffer.Guarded(after, 512, 0xa5, 64, 128));
        Check(!Native.GpuBuffer.Guarded(clean, 512, 0xa5, 500, 20));
        Check(!Oracle.Strict(Oracle.Compare(new[] { 0f, 1f }, new[] { -0f, 1f }, 1, 2)));
        Check(Oracle.Strict(Oracle.Compare(new[] { float.NaN, 1f }, new[] { float.NaN, 1f }, 1, 2)));
        Check(!Oracle.Strict(Oracle.Compare(new[] { BitConverter.UInt32BitsToSingle(0x7fc00001), 1f }, new[] { BitConverter.UInt32BitsToSingle(0x7fc00002), 1f }, 1, 2)));
        var plan = Native.MakeRangePlan(3, 17, 23); Check(plan != null);
        var assembly = typeof(Q35CudaEngine).Assembly; var chunk = typeof(Q35CudaEngine).GetNestedType("Q81RowRange", BindingFlags.NonPublic)!;
        var range = Activator.CreateInstance(chunk, new object[] { 3, 17 })!;
        var chunks = (Array)Oracle.Method(typeof(Q35CudaEngine), "Q81Chunks", true, chunk.MakeArrayType(), chunk).Invoke(null, new[] { range })!;
        Check(chunks.Length == 3);
        Check(chunks.Cast<object>().Select(x => (int)chunk.GetProperty("Start")!.GetValue(x)!).SequenceEqual(new[] { 3, 11, 19 }));
        Check(chunks.Cast<object>().Select(x => (int)chunk.GetProperty("Rows")!.GetValue(x)!).SequenceEqual(new[] { 8, 8, 1 }));
        foreach (int n in Enumerable.Range(8, 8))
        {
            var r = Activator.CreateInstance(chunk, new object[] { 3, n })!;
            var parts = (Array)Oracle.Method(typeof(Q35CudaEngine), "Q81Chunks", true, chunk.MakeArrayType(), chunk).Invoke(null, new[] { r })!;
            var actualRows = parts.Cast<object>().SelectMany(x => Enumerable.Range((int)chunk.GetProperty("Start")!.GetValue(x)!, (int)chunk.GetProperty("Rows")!.GetValue(x)!)).ToArray();
            Check(actualRows.SequenceEqual(Enumerable.Range(3, n)));
        }
        string dir = Path.GetFullPath(Path.Combine("artifacts", "q81-self-" + Guid.NewGuid().ToString("N"))); Directory.CreateDirectory(Path.GetDirectoryName(dir)!); Oracle.Fresh(dir);
        bool rejected = false; try { Oracle.Fresh(dir); } catch (InvalidOperationException) { rejected = true; } Check(rejected);
        Directory.Delete(dir); Console.WriteLine($"Q81 oracle CPU self-tests: {count} PASS; no CUDA/model/GPU calls."); return 0;
    }
}
