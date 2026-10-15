// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text.Json.Serialization;

internal sealed class Capture
{
    [JsonPropertyName("id")] public long Id { get; set; }
    [JsonPropertyName("file")] public string File { get; set; } = "";
    [JsonPropertyName("name")] public string Name { get; set; } = "";
    [JsonPropertyName("mode")] public string Mode { get; set; } = "";
    [JsonPropertyName("model")] public ulong Model { get; set; }
    [JsonPropertyName("graph")] public ulong Graph { get; set; }
    [JsonPropertyName("evaluation")] public ulong Evaluation { get; set; }
    [JsonPropertyName("tokens")] public int Tokens { get; set; }
    [JsonPropertyName("layer")] public int Layer { get; set; }
    [JsonPropertyName("row")] public int Row { get; set; }
    [JsonPropertyName("rows")] public int Rows { get; set; }
    [JsonPropertyName("clamp")] public double Clamp { get; set; }
    [JsonPropertyName("type")] public int Type { get; set; }
    [JsonPropertyName("type_name")] public string TypeName { get; set; } = "";
    [JsonPropertyName("ne")] public long[] Ne { get; set; } = [];
    [JsonPropertyName("nb")] public long[] Nb { get; set; } = [];
    [JsonPropertyName("nbytes")] public long NBytes { get; set; }
    [JsonIgnore] public byte[] Bytes { get; private set; } = [];
    [JsonIgnore] public long Frame { get; set; }
    [JsonIgnore] public string Sha256 => Convert.ToHexString(SHA256.HashData(Bytes));

    internal void Load(string root)
    {
        if (Ne.Length != 4 || Nb.Length != 4 || Ne.Any(x => x <= 0) || Nb.Any(x => x < 0)
            || NBytes < 0 || NBytes > 8 << 20 || Path.GetFileName(File) != File)
            throw new InvalidDataException("Invalid bounded capture geometry/path.");
        Bytes = System.IO.File.ReadAllBytes(Path.Combine(root, File));
        if (Bytes.LongLength != NBytes) throw new InvalidDataException("Captured payload length mismatch.");
    }

    internal int Offset(long i0, long i1 = 0, long i2 = 0, long i3 = 0)
    {
        long[] indices = [i0, i1, i2, i3];
        long offset = 0;
        for (int d = 0; d < 4; ++d)
        {
            if (indices[d] < 0 || indices[d] >= Ne[d]) throw new InvalidDataException("Captured index outside shape.");
            offset = checked(offset + indices[d] * Nb[d]);
        }
        if (offset < 0 || offset > Bytes.Length - 4) throw new InvalidDataException("Captured stride outside byte span.");
        return checked((int)offset);
    }
    internal double F32(long i0, long i1 = 0, long i2 = 0)
    {
        if (TypeName != "f32") throw new NotSupportedException("F32 capture required.");
        float value = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(Bytes.AsSpan(Offset(i0, i1, i2), 4)));
        if (!float.IsFinite(value)) throw new ArithmeticException("Nonfinite captured F32 value.");
        return value;
    }
    internal int I32(long i0, long i1 = 0)
    {
        if (TypeName != "i32") throw new NotSupportedException("I32 routing capture required.");
        return BinaryPrimitives.ReadInt32LittleEndian(Bytes.AsSpan(Offset(i0, i1), 4));
    }

    internal double[] Matrix(int width, int rows)
    {
        if (Ne[0] != width || Ne[1] != rows || Ne[2] != 1 || Ne[3] != 1) throw new InvalidDataException("Matrix shape mismatch.");
        var values = new double[checked(width * rows)];
        for (int row = 0; row < rows; ++row) for (int k = 0; k < width; ++k) values[row * width + k] = F32(k, row);
        return values;
    }
    internal double[] Routed(int width, int used, int rows, bool compact = false)
    {
        if (Ne[0] != width || Ne[1] != (compact ? used * rows : used) || Ne[2] != (compact ? 1 : rows) || Ne[3] != 1)
            throw new InvalidDataException("Routed projection shape mismatch: " + Name);
        var values = new double[checked(width * used * rows)];
        for (int row = 0; row < rows; ++row) for (int route = 0; route < used; ++route) for (int k = 0; k < width; ++k)
            values[(row * used + route) * width + k] = compact ? F32(k, route * rows + row) : F32(k, route, row);
        return values;
    }
    internal double[] TokenRows(int width, int rows)
    {
        if (Ne[1] != 1 || Ne[2] != rows) return Matrix(width, rows);
        if (Ne[0] != width || Ne[3] != 1) throw new InvalidDataException("Bulk [H,1,N] input shape mismatch.");
        var values = new double[checked(width * rows)];
        for (int row = 0; row < rows; ++row) for (int k = 0; k < width; ++k) values[row * width + k] = F32(k, 0, row);
        return values;
    }
    internal int[] RoutingIds(int used, int rows)
    {
        if (Ne[0] != used || Ne[1] != rows || Ne[2] != 1 || Ne[3] != 1) throw new InvalidDataException("Routing shape mismatch.");
        var values = new int[checked(used * rows)];
        for (int row = 0; row < rows; ++row) for (int route = 0; route < used; ++route) values[row * used + route] = I32(route, row);
        return values;
    }

    internal static void SelfTest()
    {
        static void Check(bool value, string name) { if (!value) throw new InvalidOperationException(name); }
        var ids = new Capture { TypeName = "i32", Ne = [2, 3, 1, 1], Nb = [4, 32, 96, 96], NBytes = 72, Bytes = new byte[72] };
        int[] expected = [2, 7, 4, 6, 0, 5];
        for (int row = 0; row < 3; ++row) for (int k = 0; k < 2; ++k)
            BinaryPrimitives.WriteInt32LittleEndian(ids.Bytes.AsSpan(row * 32 + k * 4, 4), expected[row * 2 + k]);
        Check(ids.RoutingIds(2, 3).SequenceEqual(expected), "Selected IDs preserve original expert-count row stride");
        var compact = new Capture { TypeName = "f32", Ne = [2, 6, 1, 1], Nb = [4, 8, 48, 48], NBytes = 48, Bytes = new byte[48] };
        for (int k = 0; k < 2; ++k) for (int row = 0; row < 3; ++row) for (int ff = 0; ff < 2; ++ff)
            BinaryPrimitives.WriteInt32LittleEndian(compact.Bytes.AsSpan(((k * 3 + row) * 2 + ff) * 4, 4),
                BitConverter.SingleToInt32Bits(100 * row + 10 * k + ff));
        var result = compact.Routed(2, 2, 3, true);
        Check(result.SequenceEqual(new double[] { 0, 1, 10, 11, 100, 101, 110, 111, 200, 201, 210, 211 }),
            "Compact route-major channels restore [row,route,feature]");
        var bulkInput = new Capture { TypeName = "f32", Ne = [2, 1, 3, 1], Nb = [4, 8, 8, 24], NBytes = 24, Bytes = new byte[24] };
        for (int row = 0; row < 3; ++row) for (int h = 0; h < 2; ++h)
            BinaryPrimitives.WriteInt32LittleEndian(bulkInput.Bytes.AsSpan((row * 2 + h) * 4, 4),
                BitConverter.SingleToInt32Bits(100 * row + h));
        Check(bulkInput.TokenRows(2, 3).SequenceEqual(new double[] { 0, 1, 100, 101, 200, 201 }),
            "Bulk [H,1,N] input preserves original token rows");
        var bulkProjection = new Capture { TypeName = "f32", Ne = [2, 2, 3, 1], Nb = [4, 8, 16, 48], NBytes = 48, Bytes = new byte[48] };
        for (int row = 0; row < 3; ++row) for (int route = 0; route < 2; ++route) for (int ff = 0; ff < 2; ++ff)
            BinaryPrimitives.WriteInt32LittleEndian(bulkProjection.Bytes.AsSpan(((row * 2 + route) * 2 + ff) * 4, 4),
                BitConverter.SingleToInt32Bits(100 * row + 10 * route + ff));
        Check(bulkProjection.Routed(2, 2, 3).SequenceEqual(result), "Bulk [FF,K,N] restores the same canonical rows");
        try { ids.I32(2, 0); throw new Exception("Expected shape refusal"); } catch (InvalidDataException) { }
        try { compact.Matrix(2, 3); throw new Exception("Expected layout refusal"); } catch (InvalidDataException) { }
        Capture[] frames = new[] { "input", "gate_projection", "input", "gate_projection",
            "selected_boundary_input", "compact_input", "input", "selected_boundary_input", "compact_input" }
            .Select((name, i) => new Capture { Id = i + 1, Name = name, Mode = i < 4 ? "ordinary" : "compact",
                Model = 10, Graph = 42, Evaluation = 1, Layer = 1, Tokens = 16 }).ToArray();
        AssignFrames(frames);
        Check(frames.Select(x => x.Frame).SequenceEqual(new long[] { 1, 1, 2, 2, 3, 3, 3, 4, 4 }),
            "Explicit markers separate reused graph pointers/counters; compact postcompute input stays in its frame");
        try { AssignFrames([new Capture { Id = 1, Name = "gate_projection", Mode = "ordinary" }]); throw new Exception("Expected missing marker refusal"); }
        catch (InvalidDataException) { }
        try { AssignFrames([new Capture { Id = 1, Name = "input", Mode = "ordinary", Graph = 1 },
            new Capture { Id = 2, Name = "gate_projection", Mode = "ordinary", Graph = 2 }]); throw new Exception("Expected context refusal"); }
        catch (InvalidDataException) { }
        try { AssignFrames([new Capture { Id = 1, Name = "input", Mode = "ordinary" },
            new Capture { Id = 1, Name = "input", Mode = "ordinary" }]); throw new Exception("Expected duplicate ID refusal"); }
        catch (InvalidDataException) { }
        Console.WriteLine("PASS: 10 independent stride, scalar/bulk layout, shape-bounds and reused-frame/refusal checks.");
    }

    internal static void AssignFrames(Capture[] captures)
    {
        if (captures.Any(c => c.Id <= 0 || c.Mode is not ("ordinary" or "compact"))
            || captures.Select(c => c.Id).Distinct().Count() != captures.Length)
            throw new InvalidDataException("Invalid or duplicated capture identity.");
        long frame = 0;
        foreach (var capture in captures.OrderBy(c => c.Id))
        {
            if (capture.Name == (capture.Mode == "compact" ? "selected_boundary_input" : "input")) ++frame;
            if (frame == 0) throw new InvalidDataException("Capture payload precedes its explicit evaluation marker.");
            capture.Frame = frame;
        }
        if (captures.GroupBy(c => c.Frame).Any(g => g.Select(c => (c.Mode, c.Model, c.Graph, c.Evaluation, c.Layer, c.Tokens)).Distinct().Count() != 1))
            throw new InvalidDataException("Recorded context changed within one explicit capture frame.");
    }
}
