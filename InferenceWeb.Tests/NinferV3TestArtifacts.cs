using System.Buffers.Binary;
using System.Text;
using System.Text.Json.Nodes;

namespace InferenceWeb.Tests;

/// <summary>Sparse generated NInfer v3 containers: the 32-byte entry header, a JSON directory
/// and a 4096-aligned payload in which only NVFP4 divisors and FP32 scalars are written.</summary>
internal sealed class NinferV3Builder
{
    private readonly JsonArray _objects = new();
    private readonly JsonObject _bindings = new();
    private readonly JsonArray _uses = new();
    private readonly JsonObject _components = new();
    private readonly List<(ulong Offset, byte[] Bytes)> _writes = new();
    private ulong _cursor;
    private int _next;
    internal string Model { get; set; } = "qwen3.8-27b";
    internal string Recipe { get; set; } = "qwen3_8_27b_nvfp4";

    internal static ulong Bytes(string format, ulong[] shape)
    {
        ulong count = shape.Aggregate(1UL, (a, b) => a * b);
        static ulong Align(ulong v) => (v + 255) / 256 * 256;
        switch (format)
        {
            case "bf16": return count * 2;
            case "fp32": case "int32": return count * 4;
            case "fp8_e4m3fn_row_bf16": return Align(count) + shape[0] * 2;
            case "nvfp4": return Align(count / 2) + count / 16 + 4;
        }
        ulong n = shape[0], kp = (shape[1] + 127) / 128 * 128;
        var (group, low, high) = format switch
        {
            "q4_g64_fp16" => (64UL, 32UL, 0UL), "q5_g64_fp16" => (64UL, 32UL, 8UL),
            "q6_g64_fp16" => (64UL, 32UL, 16UL), "q8_g32_fp16" => (32UL, 32UL, 0UL),
            _ => throw new ArgumentException(format)
        };
        ulong groups = n * kp / group;
        return Align(groups * low) + Align(groups * high) + groups * 2;
    }

    internal static string Layout(string format) => format switch
    {
        "bf16" or "fp32" or "int32" => "contiguous_le_v1",
        "fp8_e4m3fn_row_bf16" => "row_scale_v1",
        "nvfp4" => "block_scale_k16_m128x4_v1",
        _ => "row_split_k128_v1"
    };

    internal string Tensor(string format, ulong[] shape, float? nvfp4Divisor = 1f)
    {
        string id = $"weight/{_next++:D6}";
        ulong bytes = Bytes(format, shape);
        _cursor = (_cursor + 255) / 256 * 256;
        _objects.Add(new JsonObject
        {
            ["id"] = id, ["shape"] = new JsonArray(shape.Select(x => (JsonNode)JsonValue.Create(x)!).ToArray()),
            ["format"] = format, ["layout"] = Layout(format), ["offset"] = _cursor, ["bytes"] = bytes, ["kind"] = "tensor"
        });
        if (format == "nvfp4" && nvfp4Divisor is float d)
            _writes.Add((_cursor + bytes - 4, BitConverter.GetBytes(d)));
        _cursor = (_cursor + bytes + 255) / 256 * 256;
        return id;
    }

    /// <summary>A tensor whose encoded bytes (including any NVFP4 divisor) are written as given.</summary>
    internal string TensorBytes(string format, ulong[] shape, byte[] payload)
    {
        if ((ulong)payload.Length != Bytes(format, shape)) throw new ArgumentException("Payload length differs from the format geometry.");
        string id = Tensor(format, shape, nvfp4Divisor: null);
        var row = _objects[^1]!;
        _writes.Add((row["offset"]!.GetValue<ulong>(), payload));
        return id;
    }

    internal string ScalarBits(uint bits) => Scalar(BitConverter.Int32BitsToSingle(unchecked((int)bits)));

    internal string Scalar(float value)
    {
        string id = $"auxiliary/{_next++:D6}";
        _cursor = (_cursor + 255) / 256 * 256;
        _objects.Add(new JsonObject { ["id"] = id, ["shape"] = new JsonArray(), ["format"] = "fp32", ["layout"] = "contiguous_le_v1", ["offset"] = _cursor, ["bytes"] = 4, ["kind"] = "tensor" });
        _writes.Add((_cursor, BitConverter.GetBytes(value)));
        _cursor += 256;
        return id;
    }

    internal string Resource(string component, string file, byte[] bytes)
    {
        string id = $"resource/{component}/{file}";
        _objects.Add(new JsonObject { ["id"] = id, ["encoding"] = "raw_bytes_v1", ["offset"] = _cursor, ["bytes"] = bytes.Length, ["kind"] = "resource" });
        _writes.Add((_cursor, bytes));
        _cursor += (ulong)bytes.Length;
        var c = Component(component);
        var map = (c["resources"] ??= new JsonObject()).AsObject();
        map[file] = id;
        return id;
    }

    internal JsonObject Component(string name)
    {
        if (_components[name] is JsonObject existing) return existing;
        var c = new JsonObject { ["config"] = new JsonObject { ["name"] = name } };
        _components[name] = c;
        return c;
    }

    internal void Bind(string name, string id) => _bindings[name] = new JsonObject { ["object"] = id };
    internal void BindPart(string name, string id, ulong begin, ulong end)
        => _bindings[name] = new JsonObject { ["parts"] = new JsonArray(new JsonObject { ["object"] = id, ["range"] = new JsonArray(begin, end) }) };
    internal void Use(string parameter, string policy, string? divisor = null)
    {
        var use = new JsonObject { ["parameter"] = parameter, ["input"] = parameter + "/input", ["activation_policy"] = policy };
        if (divisor != null) use["auxiliaries"] = new JsonObject { ["activation_input_divisor"] = new JsonObject { ["object"] = divisor } };
        _uses.Add(use);
    }

    /// <summary>Consecutive row parts of one fused [sum(rows), columns] object.</summary>
    internal string Fused(string format, ulong columns, (string Name, ulong Rows)[] parts, string policy, float? divisor = null, float[]? divisors = null)
    {
        ulong rows = parts.Aggregate(0UL, (a, p) => a + p.Rows);
        string id = Tensor(format, new[] { rows, columns });
        ulong at = 0;
        for (int i = 0; i < parts.Length; i++)
        {
            BindPart(parts[i].Name, id, at * columns, (at + parts[i].Rows) * columns);
            at += parts[i].Rows;
            float? d = divisors?[i] ?? divisor;
            Use(parts[i].Name, policy, d is float v ? Scalar(v) : null);
        }
        return id;
    }

    internal JsonObject Directory(Func<JsonObject, JsonObject>? edit = null)
    {
        var d = new JsonObject
        {
            ["components"] = _components.DeepClone(), ["objects"] = _objects.DeepClone(), ["bindings"] = _bindings.DeepClone(),
            ["uses"] = _uses.DeepClone(), ["metadata"] = new JsonObject { ["name"] = Model },
            ["provenance"] = new JsonObject { ["converter"] = "ninfer-v3", ["recipe"] = Recipe },
            ["files"] = new JsonArray(new JsonObject { ["path"] = null, ["payload_bytes"] = _cursor })
        };
        return edit?.Invoke(d) ?? d;
    }

    internal NinferV3File Write(Func<JsonObject, JsonObject>? edit = null, string magic = "NINFER\0\x03")
        => new(Directory(edit), _writes, _cursor, magic);
}

internal sealed class NinferV3File : IDisposable
{
    private readonly string _dir = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "ts-ninfer-v3-" + Guid.NewGuid().ToString("N"));
    internal string Path { get; }
    internal long PayloadStart { get; }
    internal int JsonBytes { get; }

    internal NinferV3File(JsonObject directory, IEnumerable<(ulong Offset, byte[] Bytes)> writes, ulong payloadBytes, string magic)
    {
        System.IO.Directory.CreateDirectory(_dir);
        Path = System.IO.Path.Combine(_dir, "model.ninfer");
        byte[] json = Encoding.UTF8.GetBytes(directory.ToJsonString());
        JsonBytes = json.Length;
        PayloadStart = (32L + json.Length + 4095) / 4096 * 4096;
        using var stream = File.Create(Path);
        stream.Write(Encoding.Latin1.GetBytes(magic));
        Span<byte> header = stackalloc byte[24];
        BinaryPrimitives.WriteUInt64LittleEndian(header, (ulong)json.Length);
        for (int i = 8; i < 24; i++) header[i] = (byte)i;
        stream.Write(header);
        stream.Write(json);
        stream.SetLength(PayloadStart + (long)payloadBytes);
        foreach (var (offset, bytes) in writes)
        {
            stream.Position = PayloadStart + (long)offset;
            stream.Write(bytes);
        }
    }

    public void Dispose() => System.IO.Directory.Delete(_dir, true);
}

/// <summary>The NInfer v3 Qwen3.8-27B text + DFlash2 binding layout with generated sparse objects.</summary>
internal static class NinferV3Qwen38
{
    internal const ulong H = 5120, FF = 17408, V = 248320;

    internal static NinferV3Builder Build(int fp8MlpFrom = 56, bool draft = true, bool shortlist = true,
        Action<NinferV3Builder, int>? layerOverride = null, float upDivisor = 836f)
    {
        var b = new NinferV3Builder();
        b.Component("text");
        string emb = b.Tensor("fp8_e4m3fn_row_bf16", new[] { V, H }); b.Bind("text/token_embedding", emb);
        string head = b.Tensor("fp8_e4m3fn_row_bf16", new[] { V, H }); b.Bind("text/output_head", head);
        b.Use("text/output_head", "AllowA8");
        b.Bind("text/final_norm", b.Tensor("bf16", new[] { H }));
        for (int i = 0; i < 64; i++)
        {
            string p = $"text/layers/{i}/";
            if (layerOverride != null) { layerOverride(b, i); continue; }
            Layer(b, i, fp8MlpFrom, upDivisor);
        }
        if (draft) Draft(b, shortlist);
        return b;
    }

    internal static void Layer(NinferV3Builder b, int i, int fp8MlpFrom = 56, float upDivisor = 836f)
    {
        string p = $"text/layers/{i}/";
        b.Bind(p + "input_norm", b.Tensor("bf16", new[] { H }));
        b.Bind(p + "post_attention_norm", b.Tensor("bf16", new[] { H }));
        if ((i + 1) % 4 != 0)
        {
            b.Fused("fp8_e4m3fn_row_bf16", H, new[] { (p + "gdn/query", 2048UL), (p + "gdn/key", 2048UL), (p + "gdn/value", 6144UL), (p + "gdn/z", 6144UL) }, "AllowA8");
            b.Fused("bf16", H, new[] { (p + "gdn/a_projection", 48UL), (p + "gdn/b_projection", 48UL) }, "A16Only");
            b.Bind(p + "gdn/a_log", b.Tensor("fp32", new ulong[] { 48 }));
            b.Bind(p + "gdn/dt_bias", b.Tensor("fp32", new ulong[] { 48 }));
            b.Bind(p + "gdn/convolution", b.Tensor("bf16", new ulong[] { 4, 10240 }));
            b.Bind(p + "gdn/norm", b.Tensor("bf16", new ulong[] { 128 }));
            string o = b.Tensor("fp8_e4m3fn_row_bf16", new[] { H, 6144UL }); b.Bind(p + "gdn/output", o); b.Use(p + "gdn/output", "AllowA8");
        }
        else
        {
            b.Fused("fp8_e4m3fn_row_bf16", H, new[] { (p + "attention/query", 6144UL), (p + "attention/key", 1024UL), (p + "attention/gate", 6144UL), (p + "attention/value", 1024UL) }, "AllowA8");
            b.Bind(p + "attention/query_norm", b.Tensor("bf16", new ulong[] { 256 }));
            b.Bind(p + "attention/key_norm", b.Tensor("bf16", new ulong[] { 256 }));
            string o = b.Tensor("fp8_e4m3fn_row_bf16", new[] { H, 6144UL }); b.Bind(p + "attention/output", o); b.Use(p + "attention/output", "AllowA8");
        }
        if (i < fp8MlpFrom)
        {
            b.Fused("nvfp4", H, new[] { (p + "mlp/gate", FF), (p + "mlp/up", FF) }, "AllowA4", divisors: new[] { 836f, upDivisor });
            string d = b.Tensor("nvfp4", new[] { H, FF }); b.Bind(p + "mlp/down", d); b.Use(p + "mlp/down", "AllowA4", b.Scalar(161f));
        }
        else
        {
            b.Fused("fp8_e4m3fn_row_bf16", H, new[] { (p + "mlp/gate", FF), (p + "mlp/up", FF) }, "AllowA8");
            string d = b.Tensor("fp8_e4m3fn_row_bf16", new[] { H, FF }); b.Bind(p + "mlp/down", d); b.Use(p + "mlp/down", "AllowA8");
        }
    }

    internal static void Draft(NinferV3Builder b, bool shortlist)
    {
        b.Component("dflash2");
        string feature = b.Tensor("q8_g32_fp16", new[] { H, 25600UL }); b.Bind("dflash2/feature_projection", feature); b.Use("dflash2/feature_projection", "A16Only");
        b.Bind("dflash2/context_norm", b.Tensor("bf16", new[] { H }));
        for (int i = 0; i < 5; i++)
        {
            string p = $"dflash2/layers/{i}/";
            b.Bind(p + "input_norm", b.Tensor("bf16", new[] { H }));
            string qkv = b.Fused("q8_g32_fp16", H, new[] { (p + "attention/query", 4096UL), (p + "attention/key", 1024UL), (p + "attention/value", 1024UL) }, "A16Only");
            b.BindPart(p + "attention/context_key", qkv, 4096 * H, 5120 * H);
            b.BindPart(p + "attention/context_value", qkv, 5120 * H, 6144 * H);
            b.Bind(p + "attention/query_norm", b.Tensor("bf16", new ulong[] { 128 }));
            b.Bind(p + "attention/key_norm", b.Tensor("bf16", new ulong[] { 128 }));
            b.Bind(p + "attention/output", b.Tensor("q8_g32_fp16", new[] { H, 4096UL }));
            b.Bind(p + "attention_conv/base_kernel", b.Tensor("bf16", new ulong[] { 2, 2, H }));
            b.Bind(p + "attention_conv/kernel_projection", b.Tensor("bf16", new ulong[] { 1280, H }));
            b.Bind(p + "post_attention_norm", b.Tensor("bf16", new[] { H }));
            b.Fused("q8_g32_fp16", H, new[] { (p + "mlp/gate", FF), (p + "mlp/up", FF) }, "A16Only");
            b.Bind(p + "mlp/down", b.Tensor("q8_g32_fp16", new[] { H, FF }));
            b.Bind(p + "mlp_conv/base_kernel", b.Tensor("bf16", new ulong[] { 2, 2, H }));
            b.Bind(p + "mlp_conv/kernel_projection", b.Tensor("bf16", new ulong[] { 1280, H }));
        }
        b.Bind("dflash2/final_norm", b.Tensor("bf16", new[] { H }));
        b.Bind("dflash2/candidate_selector/hidden_projection", b.Tensor("bf16", new ulong[] { 256, H }));
        b.Bind("dflash2/candidate_selector/predecessor_codebook", b.Tensor("bf16", new ulong[] { V, 256 }));
        b.Bind("dflash2/candidate_selector/successor_codebook", b.Tensor("bf16", new ulong[] { V, 256 }));
        if (shortlist)
        {
            b.Bind("proposal/head", b.Tensor("q4_g64_fp16", new ulong[] { 131072, H }));
            b.Bind("proposal/token_ids", b.Tensor("int32", new ulong[] { 131072 }));
        }
    }
}
