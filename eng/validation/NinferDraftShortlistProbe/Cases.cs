// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Cuda;
using R = TensorSharp.Validation.NinferDraftShortlistReference;
using DraftReference = TensorSharp.Validation.NinferDFlashReference;

internal static partial class Program
{
    private static readonly int[] ActiveRows = { 0, 1, 3, 4, 15, 16, 63, 64, 127, 128, 65535, R.Rows - 1 };
    private static readonly int[] Tokens = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 21, 28, 33 };
    private static ushort F16(float x) => unchecked((ushort)BitConverter.HalfToInt16Bits((Half)x));
    private static int Encoded(int row, int column) => (row * 7 + column * 11 + column / 64) & 15;
    private static int[] Map() => Enumerable.Range(0, R.Rows).Select(i => i < 2 ? 248070 : (i * 37 + 17) % 248320).ToArray();
    private static (byte[] Codes, ushort[] Scales) Weights(string kind)
    {
        var codes = new byte[checked(R.Rows * R.Width / 2)]; var scales = new ushort[R.Rows * (R.Width / 64)];
        for (int row = 0; row < R.Rows; row++) for (int g = 0; g < 80; g++)
        {
            float scale = kind == "binary" ? new[] { .0625f, .125f, .25f }[(row + g) % 3]
                : kind == "dense-grid" ? .125f + ((row * 17 + g * 13) % 1024) / 8192f
                : kind == "dispatch-witness" ? R.Half(0x3555) : .013f + ((row * 11 + g * 23) % 67) / 171f;
            if (kind == "binary" && (row + g) % 11 == 0) scale = -scale;
            scales[row * 80 + g] = F16(scale);
        }
        foreach (int row in ActiveRows)
            if (kind == "dispatch-witness") codes[row * (R.Width / 2)] = 0x67;
            else for (int c = 0; c < R.Width; c += 2) codes[row * (R.Width / 2) + c / 2] = (byte)(Encoded(row, c) | Encoded(row, c + 1) << 4);
        return (codes, scales);
    }
    private static ushort[] Input(string kind, int tokens)
    {
        var x = new ushort[tokens * R.Width];
        for (int t = 0; t < tokens; t++) for (int c = 0; c < R.Width; c++)
        {
            float v = kind == "binary" ? (((c * 7 + t * 3) % 17) - 8) / 64f
                : kind == "dense-grid" ? ((c + t * 3) % 3 - 1) / 512f
                : kind == "dispatch-witness" ? (c > 1 ? 0 : (c == 0 ? 1 : -1) * (t % 2 == 0 ? 1 : -1))
                : (((c * 19 + t * 7) % 127) - 63) / 512f;
            x[t * R.Width + c] = R.Bf(v);
        }
        return x;
    }
    private static void ProveMmaGrid(byte[] codes, ushort[] scales, ReadOnlySpan<ushort> input, int row, double quantum)
    {
        double absoluteUnits = 0;
        for (int c = 0; c < R.Width; c++)
        {
            float w = R.Float(R.Bf(R.Q(codes, row, c) * R.Scale(scales, row, c)));
            double units = (double)w * R.Float(input[c]) * quantum;
            Need(units == Math.Truncate(units), "MMA exact-grid fixture has a fractional lattice unit.");
            absoluteUnits += Math.Abs(units);
        }
        Need(absoluteUnits < 16777216, "MMA exact-grid fixture cannot prove every partial sum exactly F32 representable.");
    }
    private static IEnumerable<Case> AllCases()
    {
        foreach (string kind in new[] { "binary", "dense-grid", "dispatch-witness", "broad-envelope" })
        {
            var w = Weights(kind); var map = Map();
            foreach (int tokens in kind == "broad-envelope" ? new[] { 7, 14, 21, 28 } : Tokens)
            {
                var input = Input(kind, tokens); var expected = new ushort[tokens * R.Rows];
                var ideal = new double[expected.Length]; var bounds = new double[expected.Length];
                bool exact = kind != "broad-envelope" || tokens <= 8;
                for (int t = 0; t < tokens; t++) foreach (int row in ActiveRows)
                {
                    var x = input.AsSpan(t * R.Width, R.Width); int at = t * R.Rows + row;
                    var reference = R.Ideal(w.Codes, w.Scales, x, row, tokens); ideal[at] = reference.Sum; bounds[at] = R.Bound(reference.Sum, reference.Absolute);
                    if (tokens == 1) expected[at] = R.GemvRow(w.Codes, w.Scales, x, row);
                    else if (tokens <= 8) expected[at] = R.SmallRowExactGrid(w.Codes, w.Scales, x, row);
                    else if (exact)
                    {
                        double quantum = kind == "binary" ? 1024 : kind == "dense-grid" ? 524288 : 64;
                        ProveMmaGrid(w.Codes, w.Scales, x, row, quantum);
                        Need((double)(float)reference.Sum == reference.Sum, "MMA exact mathematical sum is not exactly F32.");
                        expected[at] = R.Bf((float)reference.Sum);
                    }
                    if (exact) Need(Math.Abs(R.Float(expected[at]) - ideal[at]) <= bounds[at], "Exact CPU head misses its fixed FP64 bound.");
                }
                var inputs = new Dictionary<string, byte[]> { ["codes"] = w.Codes, ["scales"] = Bytes(w.Scales), ["input"] = Bytes(input) };
                var outputs = new Dictionary<string, Expected> { ["head"] = new(exact ? Bytes(expected) : null, ideal, bounds) };
                if (exact)
                {
                    var top = DraftReference.Top16(expected, R.Rows, tokens); var mapped = R.Remap(top.Ids, map);
                    inputs["map"] = Bytes(map); outputs["ids"] = new(Bytes(mapped)); outputs["unary"] = new(Bytes(top.Values));
                    outputs["workspace"] = new(null, ScratchBytes: checked((int)CudaNinferDraftModule.Top16WorkspaceBytes(R.Rows, tokens)));
                }
                yield return new($"q4-shortlist-{kind}-t{tokens}", inputs, outputs,
                    (head, draft, p) => { head.Project(p["codes"], p["scales"], p["input"], p["head"], tokens, IntPtr.Zero);
                        if (exact) { draft.Top16(p["head"], p["ids"], p["unary"], R.Rows, tokens, p["workspace"], IntPtr.Zero); head.Remap(p["ids"], p["map"], tokens * 16L, IntPtr.Zero); } },
                    new { format = "original Q4G64_F16S adjacent signed-nibble codes + row-major F16 G64 scales", tokens, originalRoute = CudaNinferDraftShortlistModule.Plan(tokens).Route,
                        exactBits = exact, activeRows = ActiveRows, allOtherCodeRows = "exact zero codes; complete outputs checked", actualArtifactWeights = false,
                        bound = "(1+2^-8)*gamma_(5120+128)*sum(abs(original-route products))+2^-8*abs(original-route FP64ideal); u=2^-24",
                        smallRoute = "integer-Q4/BF16 exact64-dot then F16scale FMA;8split pair+serial reduction", largeRoute = "BF16-rounded scaledweights before original BF16 MMA",
                        pipeline = exact ? "Project→Top16(131072)→original row-to-global map; unary exact F32 widening, duplicates retained" : "NOT_RUN for this bounded-projection fixture; no top16/mapping pass inferred", workspace = "guards only" });
            }
        }
        foreach (int count in new[] { 1, 17, 255, 256, 529 })
        {
            int[] map = Map(), ids = Enumerable.Range(0, count).Select(i => new[] { -1, int.MinValue, 0, 1, R.Rows - 1, R.Rows, int.MaxValue, 128 }[i % 8]).ToArray();
            yield return new($"remap-range-n{count}", new() { ["map"] = Bytes(map), ["ids"] = Bytes(ids) }, new() { ["ids"] = new(Bytes(R.Remap(ids, map))) },
                (head, _, p) => head.Remap(p["ids"], p["map"], count, IntPtr.Zero),
                new { oracle = "Original I32 row-range behavior: invalid rows unchanged; valid rows replaced verbatim; duplicate global tokens retained", exactBits = true, projection = "NOT_RUN; standalone map primitive" });
        }
    }
    private static void SelfTest(bool all)
    {
        for (int q = 0; q < 16; q++) Need(R.Code((byte)q, 0) == (q < 8 ? q : q - 16) && R.Code((byte)(q << 4), 1) == (q < 8 ? q : q - 16), "Signed adjacent code decode differs.");
        for (int t = 1; t <= 33; t++) { var p = CudaNinferDraftShortlistModule.Plan(t); Need(p.Index == (t == 1 ? 0 : t <= 8 ? t - 1 : 8), "Original dispatch boundary differs."); }
        byte[] codes = new byte[R.Width / 2]; codes[0] = 0x67; ushort[] scales = Enumerable.Repeat((ushort)0x3555, 80).ToArray(), input = Input("dispatch-witness", 1);
        ushort gemv = R.GemvRow(codes, scales, input, 0), small = R.SmallRowExactGrid(codes, scales, input, 0), large = R.Bf((float)R.Ideal(codes, scales, input, 0, 9).Sum);
        Need(gemv == R.Bf(R.Half(0x3555)) && small == gemv && large == R.Bf(.328125f) && large != small, "Original T8/T9 scale-boundary witness differs.");
        int built = 0; long bounded = 0;
        if (all) foreach (var c in AllCases())
        {
            built++;
            foreach (var e in c.Outputs.Values) if (e.Ideal != null)
            {
                Need(e.Bounds != null && e.Bounds.Length == e.Ideal.Length, "Bound shape differs.");
                for (int i = 0; i < e.Ideal.Length; i++) Need(double.IsFinite(e.Ideal[i]) && double.IsFinite(e.Bounds![i]) && e.Bounds[i] >= 0, "Nonfinite CPU ideal/bound.");
                bounded += e.Ideal.Length;
            }
        }
        if (all) Need(built == 48, "Full shortlist fixture count differs.");
        Console.WriteLine(System.Text.Json.JsonSerializer.Serialize(new { CPU = "PASS", dispatchPlans = 33, originalScaleBoundaryWitness = new { gemvBf16 = gemv.ToString("x4"), smallBf16 = small.ToString("x4"), largeBf16 = large.ToString("x4") }, constructedCases = built, boundedOracleValues = bounded, GPU = "NOT_RUN", modelQuality = "NOT_RUN", performance = "NOT_RUN" }));
    }
}
