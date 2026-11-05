// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using TensorSharp.Runtime;

namespace TensorSharp.Models;

/// <summary>
/// The NInfer v3 Qwen3.8 artifact (qwen3_8_27b_nvfp4.ninfer) as a llama.cpp-layout <c>qwen35</c> GGUF,
/// so the ggml backends (ggml_metal, ggml_cuda, ggml_vulkan, CPU) and MLX run it: the native CUDA engine
/// reads the artifact directly, those backends read GGUF.
///
/// Matrices are written as Q8_0 (ggml-metal and MLX have no FP8 type, and ggml-metal no NVFP4
/// matmul); the 8-bit codes hold the artifact's E4M3 and E2M1-x-E4M3 values almost exactly. Vectors
/// are F32. The mapping, checked against the same model's Q8_0 GGUF (correlation 0.9996-1.0 per tensor):
///   * norms (input, post-attention, final, Q/K, MTP) store w where GGUF stores 1 + w; the GDN norm is unchanged;
///   * attention Q: per head the 256 query rows, then that head's 256 gate rows;
///   * GDN value heads: GGUF head i is artifact head (i % 16) * 3 + i / 16 (value rows of attn_qkv, z,
///     alpha/beta rows, conv value channels, ssm_out columns, A and dt_bias); ssm_a = -exp(A_log);
///   * the conv kernel [4][channels] is stored [channels][4];
///   * MTP (block 64): nextn.eh_proj / enorm / hnorm / shared_head_norm plus an attention block.
/// The tokenizer metadata follows llama.cpp's GPT-2 BPE conversion of the artifact's tokenizer.json
/// (added tokens: special or &lt;|...|&gt; = control, others user-defined; missing ids [PADn], unused).
/// </summary>
public static class NinferQwen38GgufConverter
{
    public const string SourceKey = "tensorsharp.source.ninfer_artifact_id";
    public const string FormatKey = "tensorsharp.source.ninfer_conversion";
    public const string FormatVersion = "qwen35-q8_0-v1";
    public const string DirectoryEnvironment = "TS_NINFER_GGUF_DIR";
    private const int ValueHeads = 48, KeyHeads = 16, GdnHeadDim = 128, AttnHeadDim = 256;

    /// <summary>Where the converted GGUF lives: beside the artifact, or under TS_NINFER_GGUF_DIR.</summary>
    public static string TargetPath(string ninferPath)
    {
        string dir = Environment.GetEnvironmentVariable(DirectoryEnvironment);
        string name = Path.GetFileName(ninferPath) + ".q8_0.gguf";
        return string.IsNullOrWhiteSpace(dir) ? ninferPath + ".q8_0.gguf" : Path.Combine(dir, name);
    }

    /// <summary>The GGUF for <paramref name="ninferPath"/>, converting once (the file records the
    /// artifact id; a different artifact at the same path is converted again).</summary>
    public static string EnsureGguf(string ninferPath)
    {
        using var reader = new NinferArtifactReader(ninferPath);
        string target = TargetPath(ninferPath);
        if (File.Exists(target) && IsCurrent(target, reader.ArtifactId))
        {
            Console.Error.WriteLine($"[ninfer] using the converted GGUF {target}");
            return target;
        }
        Console.Error.WriteLine($"[ninfer] converting {ninferPath} to {target} (Q8_0 matrices, once per artifact) ...");
        var sw = Stopwatch.StartNew();
        string part = target + ".part";
        Convert(reader, part);
        File.Move(part, target, overwrite: true);
        Console.Error.WriteLine($"[ninfer] wrote {new FileInfo(target).Length >> 20} MiB in {sw.Elapsed.TotalSeconds:F0} s");
        return target;
    }

    private static bool IsCurrent(string target, string artifactId)
    {
        try
        {
            using var g = new GgufFile(target);
            return g.Metadata.TryGetValue(SourceKey, out var id) && id as string == artifactId
                && g.Metadata.TryGetValue(FormatKey, out var f) && f as string == FormatVersion;
        }
        catch (Exception) { return false; }
    }

    /// <summary>Writes the GGUF of an open artifact to <paramref name="path"/>.</summary>
    public static void Convert(NinferArtifactReader reader, string path)
    {
        var text = ReadConfig(reader);
        int layers = text.GetProperty("num_hidden_layers").GetInt32();
        int hidden = text.GetProperty("hidden_size").GetInt32();
        int vocab = text.GetProperty("vocab_size").GetInt32();
        int ff = text.GetProperty("intermediate_size").GetInt32();
        if (text.GetProperty("linear_num_value_heads").GetInt32() != ValueHeads || text.GetProperty("linear_num_key_heads").GetInt32() != KeyHeads
            || text.GetProperty("linear_value_head_dim").GetInt32() != GdnHeadDim || text.GetProperty("head_dim").GetInt32() != AttnHeadDim)
            throw new NotSupportedException("The NInfer GGUF conversion is defined for Qwen3.8-27B's geometry (48/16 GDN heads of 128, attention heads of 256).");
        var rope = text.GetProperty("rope_parameters");
        bool mtp = reader.Bindings.ContainsKey("mtp/input_projection");
        var types = text.GetProperty("layer_types").EnumerateArray().Select(t => t.GetString()).ToArray();

        var w = new GgufWriter();
        w.Add("general.architecture", "qwen35");
        w.Add("general.type", "model");
        w.Add("general.name", "Qwen3.8-27B (NInfer v3, Q8_0)");
        w.Add("general.file_type", 7u);
        w.Add("general.quantization_version", 2u);
        w.Add(SourceKey, reader.ArtifactId);
        w.Add(FormatKey, FormatVersion);
        w.Add("qwen35.block_count", (uint)(layers + (mtp ? 1 : 0)));
        w.Add("qwen35.context_length", (uint)text.GetProperty("max_position_embeddings").GetInt32());
        w.Add("qwen35.embedding_length", (uint)hidden);
        w.Add("qwen35.feed_forward_length", (uint)ff);
        w.Add("qwen35.attention.head_count", (uint)text.GetProperty("num_attention_heads").GetInt32());
        w.Add("qwen35.attention.head_count_kv", (uint)text.GetProperty("num_key_value_heads").GetInt32());
        w.Add("qwen35.rope.dimension_sections", rope.GetProperty("mrope_section").EnumerateArray().Select(e => e.GetInt32()).Append(0).ToArray());
        w.Add("qwen35.rope.freq_base", (float)rope.GetProperty("rope_theta").GetDouble());
        w.Add("qwen35.attention.layer_norm_rms_epsilon", (float)text.GetProperty("rms_norm_eps").GetDouble());
        w.Add("qwen35.attention.key_length", (uint)AttnHeadDim);
        w.Add("qwen35.attention.value_length", (uint)AttnHeadDim);
        w.Add("qwen35.nextn_predict_layers", mtp ? 1u : 0u);
        w.Add("qwen35.ssm.conv_kernel", (uint)text.GetProperty("linear_conv_kernel_dim").GetInt32());
        w.Add("qwen35.ssm.state_size", (uint)GdnHeadDim);
        w.Add("qwen35.ssm.group_count", (uint)KeyHeads);
        w.Add("qwen35.ssm.time_step_rank", (uint)ValueHeads);
        w.Add("qwen35.ssm.inner_size", (uint)(ValueHeads * GdnHeadDim));
        w.Add("qwen35.full_attention_interval", (uint)(Array.IndexOf(types, "full_attention") + 1));
        w.Add("qwen35.rope.dimension_count", (uint)Math.Round(AttnHeadDim * rope.GetProperty("partial_rotary_factor").GetDouble()));
        AddTokenizer(w, reader, vocab);

        var src = new Source(reader);
        Matrix(w, src, "token_embd.weight", Plain("text/token_embedding"));
        Vector(w, "output_norm.weight", src.Values("text/final_norm"), plusOne: true);
        Matrix(w, src, "output.weight", Plain("text/output_head"));
        for (int i = 0; i < layers; i++)
        {
            string b = $"blk.{i}.", l = $"text/layers/{i}/";
            Vector(w, b + "attn_norm.weight", src.Values(l + "input_norm"), plusOne: true);
            Vector(w, b + "post_attention_norm.weight", src.Values(l + "post_attention_norm"), plusOne: true);
            if (types[i] == "full_attention")
                Attention(w, src, b, l + "attention/");
            else
                Gdn(w, src, b, l + "gdn/");
            Mlp(w, src, b, l + "mlp/");
        }
        if (mtp)
        {
            string b = $"blk.{layers}.", m = "mtp/";
            Matrix(w, src, b + "nextn.eh_proj.weight", Plain(m + "input_projection"));
            Vector(w, b + "nextn.enorm.weight", src.Values(m + "embedding_norm"), plusOne: true);
            Vector(w, b + "nextn.hnorm.weight", src.Values(m + "hidden_norm"), plusOne: true);
            Vector(w, b + "nextn.shared_head_norm.weight", src.Values(m + "final_norm"), plusOne: true);
            Vector(w, b + "attn_norm.weight", src.Values(m + "layers/0/input_norm"), plusOne: true);
            Vector(w, b + "post_attention_norm.weight", src.Values(m + "layers/0/post_attention_norm"), plusOne: true);
            Attention(w, src, b, m + "layers/0/attention/");
            Mlp(w, src, b, m + "layers/0/mlp/");
        }
        w.WriteTo(path);
    }

    private static JsonElement ReadConfig(NinferArtifactReader reader)
    {
        if (!reader.ComponentConfigs.TryGetValue("text", out string json))
            throw new InvalidDataException("The NInfer artifact has no text component config.");
        return JsonDocument.Parse(json).RootElement.Clone();
    }

    // ---- tensors -------------------------------------------------------------------------------

    /// <summary>One output row: the binding it comes from and the binding row.</summary>
    private delegate (string Binding, int Row) RowMap(int row);

    private static (RowMap Map, int Rows, string First) Plain(string binding) => ((r => (binding, r)), -1, binding);

    private static int GdnHead(int i) => i % KeyHeads * 3 + i / KeyHeads;

    private static void Attention(GgufWriter w, Source src, string b, string a)
    {
        int heads = src.Rows(a + "query") / AttnHeadDim;
        Matrix(w, src, b + "attn_q.weight", ((r =>
        {
            int h = r / (2 * AttnHeadDim), x = r % (2 * AttnHeadDim);
            return x < AttnHeadDim ? (a + "query", h * AttnHeadDim + x) : (a + "gate", h * AttnHeadDim + x - AttnHeadDim);
        }), heads * 2 * AttnHeadDim, a + "query"));
        Matrix(w, src, b + "attn_k.weight", Plain(a + "key"));
        Matrix(w, src, b + "attn_v.weight", Plain(a + "value"));
        Matrix(w, src, b + "attn_output.weight", Plain(a + "output"));
        Vector(w, b + "attn_q_norm.weight", src.Values(a + "query_norm"), plusOne: true);
        Vector(w, b + "attn_k_norm.weight", src.Values(a + "key_norm"), plusOne: true);
    }

    private static void Gdn(GgufWriter w, Source src, string b, string g)
    {
        int qRows = src.Rows(g + "query"), kRows = src.Rows(g + "key"), vRows = src.Rows(g + "value");
        RowMap valueRow = r => (g + "value", GdnHead(r / GdnHeadDim) * GdnHeadDim + r % GdnHeadDim);
        Matrix(w, src, b + "attn_qkv.weight", ((r => r < qRows ? (g + "query", r) : r < qRows + kRows ? (g + "key", r - qRows) : valueRow(r - qRows - kRows)),
            qRows + kRows + vRows, g + "query"));
        Matrix(w, src, b + "attn_gate.weight", ((r => (g + "z", GdnHead(r / GdnHeadDim) * GdnHeadDim + r % GdnHeadDim)), vRows, g + "z"));
        Matrix(w, src, b + "ssm_alpha.weight", ((r => (g + "a_projection", GdnHead(r))), ValueHeads, g + "a_projection"));
        Matrix(w, src, b + "ssm_beta.weight", ((r => (g + "b_projection", GdnHead(r))), ValueHeads, g + "b_projection"));
        // ssm_out: the artifact's input columns follow its value-head order.
        int cols = src.Columns(g + "output");
        var colMap = new int[cols];
        for (int c = 0; c < cols; c++) colMap[c] = GdnHead(c / GdnHeadDim) * GdnHeadDim + c % GdnHeadDim;
        Matrix(w, src, b + "ssm_out.weight", Plain(g + "output"), colMap);
        float[] aLog = src.Values(g + "a_log"), dt = src.Values(g + "dt_bias");
        Vector(w, b + "ssm_a", Enumerable.Range(0, ValueHeads).Select(i => -MathF.Exp(aLog[GdnHead(i)])).ToArray(), plusOne: false);
        Vector(w, b + "ssm_dt.bias", Enumerable.Range(0, ValueHeads).Select(i => dt[GdnHead(i)]).ToArray(), plusOne: false);
        // conv [taps][channels] -> [channels][taps], value channels in GGUF head order.
        float[] conv = src.Values(g + "convolution");
        int taps = src.Rows(g + "convolution"), channels = conv.Length / taps, qk = qRows + kRows;
        var cw = new float[conv.Length];
        for (int c = 0; c < channels; c++)
        {
            int s = c < qk ? c : qk + GdnHead((c - qk) / GdnHeadDim) * GdnHeadDim + (c - qk) % GdnHeadDim;
            for (int t = 0; t < taps; t++) cw[c * taps + t] = conv[t * channels + s];
        }
        w.AddTensor(b + "ssm_conv1d.weight", new[] { (ulong)taps, (ulong)channels }, GgufWriter.TypeF32, cw.Length * 4L, s => WriteF32(s, cw));
        Vector(w, b + "ssm_norm.weight", src.Values(g + "norm"), plusOne: false);
    }

    private static void Mlp(GgufWriter w, Source src, string b, string m)
    {
        Matrix(w, src, b + "ffn_gate.weight", Plain(m + "gate"));
        Matrix(w, src, b + "ffn_up.weight", Plain(m + "up"));
        Matrix(w, src, b + "ffn_down.weight", Plain(m + "down"));
    }

    private static void Vector(GgufWriter w, string name, float[] values, bool plusOne)
    {
        var v = plusOne ? values.Select(x => x + 1.0f).ToArray() : values;
        w.AddTensor(name, new[] { (ulong)v.Length }, GgufWriter.TypeF32, v.Length * 4L, s => WriteF32(s, v));
    }

    private static void WriteF32(Stream s, float[] v)
    {
        var bytes = new byte[v.Length * 4];
        Buffer.BlockCopy(v, 0, bytes, 0, bytes.Length);
        s.Write(bytes);
    }

    /// <summary>A Q8_0 matrix of output rows mapped onto artifact bindings (optionally a column gather).</summary>
    private static void Matrix(GgufWriter w, Source src, string name, (RowMap Map, int Rows, string First) spec, int[] columns = null)
    {
        int cols = src.Columns(spec.First);
        int rows = spec.Rows >= 0 ? spec.Rows : src.Rows(spec.First);
        w.AddTensor(name, new[] { (ulong)cols, (ulong)rows }, GgufWriter.TypeQ8_0, GgufWriter.Q8_0Bytes(rows, cols), s =>
        {
            const int Chunk = 256;
            var values = new float[Chunk * cols];
            var gathered = columns == null ? null : new float[cols];
            var block = new byte[Chunk * (cols / 32) * 34];
            for (int r0 = 0; r0 < rows; r0 += Chunk)
            {
                int n = Math.Min(Chunk, rows - r0);
                // Runs of consecutive rows of one binding decode together.
                for (int i = 0; i < n;)
                {
                    var (binding, row) = spec.Map(r0 + i);
                    int run = 1;
                    while (i + run < n && spec.Map(r0 + i + run) is var (b2, rr) && b2 == binding && rr == row + run) run++;
                    src.Decode(binding, row, run, values.AsSpan(i * cols, run * cols));
                    i += run;
                }
                Parallel.For(0, n, r =>
                {
                    var rowValues = values.AsSpan(r * cols, cols);
                    if (columns != null)
                    {
                        var tmp = new float[cols];
                        for (int c = 0; c < cols; c++) tmp[c] = rowValues[columns[c]];
                        tmp.CopyTo(rowValues);
                    }
                    QuantizeQ8_0(rowValues, block.AsSpan(r * (cols / 32) * 34, (cols / 32) * 34));
                });
                s.Write(block, 0, n * (cols / 32) * 34);
            }
        });
    }

    /// <summary>ggml's quantize_row_q8_0_ref: d = amax / 127 (F16), q = round(x / d), half away from zero.</summary>
    internal static void QuantizeQ8_0(ReadOnlySpan<float> x, Span<byte> y)
    {
        for (int b = 0; b < x.Length / 32; b++)
        {
            var v = x.Slice(b * 32, 32);
            float amax = 0;
            foreach (float f in v) amax = MathF.Max(amax, MathF.Abs(f));
            float d = amax / 127.0f, id = d != 0 ? 1.0f / d : 0.0f;
            var o = y.Slice(b * 34, 34);
            BinaryPrimitives.WriteHalfLittleEndian(o, (System.Half)d);
            for (int i = 0; i < 32; i++) o[2 + i] = unchecked((byte)(sbyte)MathF.Round(v[i] * id, MidpointRounding.AwayFromZero));
        }
    }

    // ---- artifact decoding ---------------------------------------------------------------------

    private sealed class Source
    {
        private static readonly float[] E4M3 = BuildE4M3();
        private static readonly float[] E2M1 = { 0, .5f, 1, 1.5f, 2, 3, 4, 6, -0f, -.5f, -1, -1.5f, -2, -3, -4, -6 };
        private readonly NinferArtifactReader _r;
        public Source(NinferArtifactReader r) => _r = r;

        private static float[] BuildE4M3()
        {
            var t = new float[256];
            for (int b = 0; b < 256; b++)
            {
                int e = (b >> 3) & 15, m = b & 7;
                float v = e == 15 && m == 7 ? float.NaN : e == 0 ? m / 8f * MathF.Pow(2, -6) : (1 + m / 8f) * MathF.Pow(2, e - 7);
                t[b] = (b & 0x80) != 0 ? -v : v;
            }
            return t;
        }

        private NinferBinding Binding(string name) => _r.GetBinding(name);
        public int Rows(string name) => Binding(name).RowRange.Rows;
        public int Columns(string name) => checked((int)Binding(name).Object.Shape[^1]);

        /// <summary>Every value of a small binding (vectors, the conv kernel).</summary>
        public float[] Values(string name)
        {
            var b = Binding(name);
            int rows = b.Object.Shape.Count == 1 ? 1 : b.RowRange.Rows;
            int cols = b.Object.Shape.Count == 1 ? checked((int)b.Object.Shape[0]) : Columns(name);
            var v = new float[rows * cols];
            DecodeRows(b.Object, b.Object.Shape.Count == 1 ? 0 : b.RowRange.FirstRow, rows, cols, v);
            return v;
        }

        /// <summary>Rows [row, row + count) of a binding as F32.</summary>
        public void Decode(string name, int row, int count, Span<float> dst)
        {
            var b = Binding(name);
            if (row < 0 || row + count > b.RowRange.Rows) throw new ArgumentOutOfRangeException(nameof(row));
            DecodeRows(b.Object, b.RowRange.FirstRow + row, count, Columns(name), dst);
        }

        private void DecodeRows(NinferTensorDescriptor t, int first, int count, int cols, Span<float> dst)
        {
            switch (t.Format)
            {
                case NinferTensorFormat.BFloat16:
                {
                    var raw = Read(t, (ulong)first * (ulong)cols * 2, count * cols * 2);
                    for (int i = 0; i < count * cols; i++) dst[i] = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadUInt16LittleEndian(raw.AsSpan(2 * i)) << 16);
                    break;
                }
                case NinferTensorFormat.Float32:
                {
                    var raw = Read(t, (ulong)first * (ulong)cols * 4, count * cols * 4);
                    for (int i = 0; i < count * cols; i++) dst[i] = BinaryPrimitives.ReadSingleLittleEndian(raw.AsSpan(4 * i));
                    break;
                }
                case NinferTensorFormat.Fp8E4M3FnRowBFloat16Scale:
                {
                    var codes = Read(t, (ulong)first * (ulong)cols, count * cols);
                    var scales = Read(t, t.Geometry.ScalePlaneOffset!.Value + (ulong)first * 2, count * 2);
                    for (int r = 0; r < count; r++)
                    {
                        float s = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadUInt16LittleEndian(scales.AsSpan(2 * r)) << 16);
                        for (int c = 0; c < cols; c++) dst[r * cols + c] = E4M3[codes[r * cols + c]] * s;
                    }
                    break;
                }
                case NinferTensorFormat.W8G32Float16Scale:
                {
                    if (t.Geometry.PaddedK != (ulong)cols) throw new NotSupportedException("Padded q8_g32 rows are not converted.");
                    var codes = Read(t, (ulong)first * (ulong)cols, count * cols);
                    var scales = Read(t, t.Geometry.ScalePlaneOffset!.Value + (ulong)first * (ulong)(cols / 32) * 2, count * cols / 32 * 2);
                    for (int r = 0; r < count; r++)
                        for (int c = 0; c < cols; c++)
                            dst[r * cols + c] = (sbyte)codes[r * cols + c] * (float)BinaryPrimitives.ReadHalfLittleEndian(scales.AsSpan(2 * (r * (cols / 32) + c / 32)));
                    break;
                }
                case NinferTensorFormat.NvFp4:
                {
                    // Codes: adjacent-K nibbles (low = even K). Scales: NVIDIA's 128x4 tiles, one E4M3 per 16 values.
                    var codes = Read(t, (ulong)first * (ulong)cols / 2, count * cols / 2);
                    int tileBytes = cols / 64 * 512, t0 = first / 128, t1 = (first + count - 1) / 128;
                    var scales = Read(t, t.Geometry.ScalePlaneOffset!.Value + (ulong)t0 * (ulong)tileBytes, (t1 - t0 + 1) * tileBytes);
                    float inv = 1.0f / t.WeightDivisor!.Value;
                    for (int r = 0; r < count; r++)
                    {
                        int row = first + r, m = row % 128, tile = row / 128 - t0;
                        for (int gi = 0; gi < cols / 16; gi++)
                        {
                            float s = E4M3[scales[tile * tileBytes + gi / 4 * 512 + m % 32 * 16 + m / 32 * 4 + gi % 4]] * inv;
                            for (int j = 0; j < 8; j++)
                            {
                                byte b = codes[r * cols / 2 + gi * 8 + j];
                                dst[r * cols + gi * 16 + 2 * j] = E2M1[b & 15] * s;
                                dst[r * cols + gi * 16 + 2 * j + 1] = E2M1[b >> 4] * s;
                            }
                        }
                    }
                    break;
                }
                default:
                    throw new NotSupportedException($"NInfer {t.Format} tensors are not converted.");
            }
        }

        private byte[] Read(NinferTensorDescriptor t, ulong offset, int bytes)
        {
            var b = new byte[bytes];
            _r.ReadObjectRange(t, offset, b);
            return b;
        }
    }

    // ---- tokenizer -----------------------------------------------------------------------------

    private static void AddTokenizer(GgufWriter w, NinferArtifactReader reader, int vocab)
    {
        var frontend = NinferQwen38Frontend.FromReader(reader);
        using var tok = JsonDocument.Parse(frontend.Resources["frontend/tokenizer.json"].CopyBytes());
        using var cfg = JsonDocument.Parse(frontend.Resources["frontend/tokenizer_config.json"].CopyBytes());
        var model = tok.RootElement.GetProperty("model");
        var tokens = new string[vocab];
        var types = new int[vocab];
        foreach (var e in model.GetProperty("vocab").EnumerateObject())
        {
            int id = e.Value.GetInt32();
            if (id < vocab) { tokens[id] = e.Name; types[id] = 1; }
        }
        foreach (var a in tok.RootElement.GetProperty("added_tokens").EnumerateArray())
        {
            int id = a.GetProperty("id").GetInt32();
            if (id >= vocab) continue;
            string content = a.GetProperty("content").GetString();
            bool special = a.GetProperty("special").GetBoolean() || (content.StartsWith("<|") && content.EndsWith("|>"));
            tokens[id] = content;
            types[id] = special ? 3 : 4;
        }
        for (int i = 0; i < vocab; i++)
            if (tokens[i] == null) { tokens[i] = $"[PAD{i}]"; types[i] = 5; }
        var merges = model.GetProperty("merges").EnumerateArray()
            .Select(m => m.ValueKind == JsonValueKind.String ? m.GetString() : string.Join(' ', m.EnumerateArray().Select(x => x.GetString())))
            .ToArray();
        int Id(string content) => Array.IndexOf(tokens, content);
        string Special(string key) => cfg.RootElement.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.String ? v.GetString() : null;
        w.Add("tokenizer.ggml.model", "gpt2");
        w.Add("tokenizer.ggml.pre", "qwen35");
        w.Add("tokenizer.ggml.tokens", tokens);
        w.Add("tokenizer.ggml.token_type", types);
        w.Add("tokenizer.ggml.merges", merges);
        w.Add("tokenizer.ggml.eos_token_id", (uint)Id(Special("eos_token") ?? "<|im_end|>"));
        w.Add("tokenizer.ggml.padding_token_id", (uint)Id(Special("pad_token") ?? "<|endoftext|>"));
        w.Add("tokenizer.ggml.bos_token_id", (uint)Id(Special("bos_token") ?? "<|endoftext|>"));
        w.Add("tokenizer.chat_template", frontend.ChatTemplate);
    }
}
