// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The Inkling 2 test fixtures themselves: the synthetic GGUFs have the real checkpoint's layout, and the double-precision
// reference forward (written from llama.cpp's graph) is self-consistent - a sequence read in chunks with the carried state
// equals the sequence read at once - and reaches every mechanism. No backend, no model file, no GPU: these run in the
// environment-independent lane (no Requires trait).
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

public sealed class Inkling2ReferenceTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-inkling2-reference-" + Guid.NewGuid().ToString("N"));
    private readonly Xunit.Abstractions.ITestOutputHelper _output;

    /// <summary>24 positions: past the window (6), the global extent (8) and the log-scale floor (4).</summary>
    private static readonly int[] Prompt = Enumerable.Range(0, 24).Select(i => (7 + 13 * i) % 61).ToArray();

    public Inkling2ReferenceTests(Xunit.Abstractions.ITestOutputHelper output)
    {
        _output = output;
        Directory.CreateDirectory(_dir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private static Inkling2SyntheticModelBuilder Builder(bool looped, int passes = 3)
        => new() { Looped = looped, LoopCount = passes, LoopVarying = passes < 2 };

    private string Fixture(bool looped, int passes = 3)
        => Builder(looped, passes).Write(Path.Combine(_dir, $"inkling2-{(looped ? "loop" + passes : "plain")}.gguf"));

    private Inkling2Reference Reference(bool looped, int passes = 3) => Inkling2Reference.Load(Fixture(looped, passes));

    private static double MaxAbsDiff(double[][] a, double[][] b)
    {
        Assert.Equal(a.Length, b.Length);
        double max = 0;
        for (int t = 0; t < a.Length; t++)
        {
            Assert.Equal(a[t].Length, b[t].Length);
            for (int v = 0; v < a[t].Length; v++) max = Math.Max(max, Math.Abs(a[t][v] - b[t][v]));
        }
        return max;
    }

    private static double RelL2(double[] a, double[] b)
    {
        double num = 0, den = 0;
        for (int i = 0; i < a.Length; i++)
        {
            num += (a[i] - b[i]) * (a[i] - b[i]);
            den += b[i] * b[i];
        }
        return Math.Sqrt(num / Math.Max(den, 1e-300));
    }

    // ---------------------------------------------------------------- the files

    [Fact]
    public void TheSyntheticGgufHasTheRealCheckpointsLayout()
    {
        var builder = Builder(looped: true);
        string path = Fixture(looped: true);
        using var file = new GgufFile(path);
        const string a = "inkling2_loop";
        Assert.Equal(a, file.GetString("general.architecture"));
        Assert.Equal("gpt2", file.GetString("tokenizer.ggml.model"));
        Assert.Equal("inkling2", file.GetString("tokenizer.ggml.pre"));
        Assert.Equal(builder.Vocab, file.GetStringArray("tokenizer.ggml.tokens").Length);

        // value kinds as convert_hf_to_gguf writes them: counts U32, per-layer heads INT32 arrays, the pattern a BOOL array
        foreach (string key in new[] { "block_count", "embedding_length", "feed_forward_length", "expert_feed_forward_length",
                     "attention.key_length", "attention.value_length", "expert_count", "expert_used_count", "vocab_size",
                     "vocab_size_unpadded", "rope.dimension_count", "attention.sliding_window", "shortconv.l_cache",
                     "attn_res.block_size", "attention.relative_length", "attention.relative_extent", "attention.log_scale_floor",
                     "loop.start", "loop.end", "loop.count", "loop.gate_rank" })
            Assert.IsType<uint>(file.Metadata[$"{a}.{key}"]);
        foreach (string key in new[] { "attention.layer_norm_rms_epsilon", "expert_weights_scale", "activation.situ_beta",
                     "activation.situ_linear_beta", "final_logit_softcapping", "attention.qk_clip", "attention.relative_clip",
                     "attention.log_scale_alpha" })
            Assert.IsType<float>(file.Metadata[$"{a}.{key}"]);
        Assert.IsType<bool>(file.Metadata[$"{a}.loop.varying"]);
        Assert.Equal(Enumerable.Repeat(builder.Heads, builder.Layers), file.GetInt32Array($"{a}.attention.head_count"));
        Assert.Equal(Enumerable.Repeat(builder.KvHeads, builder.Layers), file.GetInt32Array($"{a}.attention.head_count_kv"));
        Assert.Equal(new[] { true, false, true, false, true, false, true, false }, file.GetBoolArray($"{a}.attention.sliding_window_pattern"));
        Assert.Equal(0u, file.GetUint32($"{a}.rope.dimension_count"));
        Assert.Equal(61u, file.GetUint32($"{a}.vocab_size_unpadded"));
        Assert.Equal(64u, file.GetUint32($"{a}.vocab_size"));

        // ne0-first shapes of the mechanisms that differ by layer kind
        ulong[] Shape(string name) => file.Tensors[name].Shape;
        Assert.Equal(new ulong[] { 64, 64 }, Shape("token_embd.weight"));
        Assert.Equal(new ulong[] { 64, 64 }, Shape("output.weight"));
        Assert.Equal(new ulong[] { 4, 6 }, Shape("blk.0.attn_rel_bias.weight"));    // sliding: extent = window
        Assert.Equal(new ulong[] { 4, 8 }, Shape("blk.1.attn_rel_bias.weight"));    // global: relative_extent
        Assert.False(file.Tensors.ContainsKey("blk.0.attn_res_query.weight"));
        Assert.Equal(new ulong[] { 64 }, Shape("blk.2.attn_res_query.weight"));
        Assert.Equal(new ulong[] { 64, 3 }, Shape("blk.3.attn_res_query.weight"));
        Assert.Equal(new ulong[] { 64, 8 }, Shape("blk.3.attn_k_conv_proj.weight"));
        Assert.False(file.Tensors.ContainsKey("blk.2.attn_k_conv_proj.weight"));
        Assert.False(file.Tensors.ContainsKey("blk.3.attn_norm.weight"));
        Assert.Equal(new ulong[] { 16, 4 }, Shape("blk.3.attn_q_clip.weight"));
        Assert.Equal(new ulong[] { 4, 32 }, Shape("blk.3.attn_k_conv.weight"));
        Assert.Equal(new ulong[] { 64, 32, 8 }, Shape("blk.3.ffn_gate_up_exps.weight"));
        Assert.Equal(new ulong[] { 16, 64, 8 }, Shape("blk.3.ffn_down_exps.weight"));
        Assert.Equal(new ulong[] { 8 }, Shape("blk.3.exp_probs_b.bias"));
        for (int l = 0; l < builder.Layers; l++)
            Assert.Equal(l >= 2 && l < 6, file.Tensors.ContainsKey($"blk.{l}.loop_msg_query.weight"));
        Assert.Equal(new ulong[] { 64, 4 }, Shape("blk.2.loop_msg_query.weight"));
        Assert.Equal(new ulong[] { 128, 8 }, Shape("blk.5.loop_ffn_ctrl_down.weight"));
        Assert.Equal(new ulong[] { 8, 64 }, Shape("blk.5.loop_v_ctrl_forget.weight"));

        // every tensor F32 with the builder's values (the routed experts' stacks F16, as the shared runtime's MoE fixtures
        // store them), and the reference's own loader accepts the whole table
        var expected = builder.Tensors().ToDictionary(t => t.Name);
        Assert.Equal(expected.Count, file.Tensors.Count);
        foreach (GgufTensorInfo info in file.Tensors.Values)
        {
            bool experts = info.Shape.Length == 3 && info.Name.Contains("_exps.", StringComparison.Ordinal);
            Assert.Equal(experts && builder.HalfExperts ? GgmlTensorType.F16 : GgmlTensorType.F32, info.Type);
            Assert.Equal(expected[info.Name].Dims, info.Shape);
            var values = new float[info.NumElements];
            byte[] raw = file.ReadTensorData(info);
            if (info.Type == GgmlTensorType.F32)
                Buffer.BlockCopy(raw, 0, values, 0, values.Length * sizeof(float));
            else
                for (int i = 0; i < values.Length; i++) values[i] = (float)BitConverter.ToHalf(raw, 2 * i);
            float[] want = info.Type == GgmlTensorType.F32 ? expected[info.Name].Data : Array.ConvertAll(expected[info.Name].Data, v => (float)(Half)v);
            Assert.Equal(want, values);
        }
        var reference = Inkling2Reference.Load(path);
        Assert.Equal(builder.Slots, reference.Cfg.Slots);
    }

    [Fact]
    public void ThePlainVariantHasNoLoop()
    {
        string path = Fixture(looped: false);
        using var file = new GgufFile(path);
        Assert.Equal("inkling2", file.GetString("general.architecture"));
        Assert.DoesNotContain(file.Metadata.Keys, k => k.Contains(".loop.", StringComparison.Ordinal));
        Assert.DoesNotContain(file.Tensors.Keys, k => k.Contains(".loop_", StringComparison.Ordinal));
        Assert.Equal(6u, file.GetUint32("inkling2.attention.sliding_window"));
        var reference = Inkling2Reference.Load(path);
        Assert.Equal(8, reference.Cfg.Slots);
        Assert.Equal(Enumerable.Range(0, 8), reference.Schedule.Select(v => v.Slot));
    }

    [Fact]
    public void TheLoopScheduleUsesLlamaCppsCacheSlots()
    {
        var reference = Reference(looped: true);
        // prelude 0-1, core 2-5 three times (later passes at slots 8 + (pass - 1) * 4 + (ip - 2)), then 6-7
        Assert.Equal(new[] { 0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 6, 7 }, reference.Schedule.Select(v => v.Slot));
        Assert.Equal(new[] { 0, 1, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 6, 7 }, reference.Schedule.Select(v => v.Physical));
        Assert.Equal(new[] { -1, -1, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, -1, -1 }, reference.Schedule.Select(v => v.Pass));
        Assert.Equal(16, reference.Cfg.Slots);
    }

    // ---------------------------------------------------------------- the reference

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void ChunkedWithCarriedStateMatchesOnePass(bool looped)
    {
        var reference = Reference(looped);
        double[][] once = reference.Logits(Prompt);
        int[][] splits =
        {
            Enumerable.Repeat(1, Prompt.Length).ToArray(),   // decode: one token at a time
            new[] { 5, 1, 7, 3, 8 },                         // chunk edges inside and across the window and the extent
            new[] { 6, 6, 6, 6 },
            new[] { 23, 1 },
        };
        foreach (int[] split in splits)
        {
            double diff = MaxAbsDiff(reference.LogitsChunked(Prompt, split), once);
            _output.WriteLine($"{(looped ? "loop" : "plain")} chunks [{string.Join(",", split)}]: max |diff| {diff:E2}");
            Assert.True(diff <= 1e-10, $"chunks [{string.Join(",", split)}]: max |diff| {diff:E2}");
        }

        // the carried state is what llama.cpp keeps: one K/V row per position in every slot, taps - 1 conv inputs
        var state = reference.NewState();
        reference.Forward(state, Prompt[..10]);
        reference.Forward(state, Prompt[10..]);
        Assert.Equal(Prompt.Length, state.Position);
        Assert.All(state.Slots, s =>
        {
            Assert.Equal(Prompt.Length, s.Keys.Count);
            Assert.Equal(Prompt.Length, s.Values.Count);
            Assert.Equal(3, s.ConvK.Length);
            Assert.Equal(3, s.ConvFfn.Length);
        });
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void TheLogitsAreFiniteUnpaddedAndCapped(bool looped)
    {
        var builder = Builder(looped);
        var reference = Reference(looped);
        double[][] logits = reference.Logits(Prompt);
        Assert.Equal(Prompt.Length, logits.Length);
        Assert.All(logits, row =>
        {
            Assert.Equal(builder.Vocab, row.Length);
            Assert.All(row, z => Assert.True(double.IsFinite(z) && Math.Abs(z) < builder.Softcap));
        });
        // the rows depend on the token and its history
        Assert.True(RelL2(logits[^1], logits[^2]) > 1e-2);
    }

    [Fact]
    public void TheLoopMatters()
    {
        double[][] three = Reference(looped: true).Logits(Prompt);
        double[][] two = Reference(looped: true, passes: 2).Logits(Prompt);
        double[][] one = Reference(looped: true, passes: 1).Logits(Prompt);
        double[] vsOne = three.Select((row, t) => RelL2(row, one[t])).ToArray();
        double[] vsTwo = three.Select((row, t) => RelL2(row, two[t])).ToArray();
        _output.WriteLine($"3 passes vs 1, relL2 per position: {string.Join(" ", vsOne.Select(r => r.ToString("E1")))}");
        _output.WriteLine($"3 passes vs 2, relL2 per position: {string.Join(" ", vsTwo.Select(r => r.ToString("E1")))}");
        Assert.All(vsOne, r => Assert.True(r > 1e-3, $"relL2 {r:E2}"));
        Assert.All(vsTwo, r => Assert.True(r > 1e-4, $"relL2 {r:E2}"));
    }

    [Fact]
    public void OnePassOfTheLoopIsThePlainModel()
    {
        // loop.count 1 runs the core once without messages: llama.cpp's inkling2 graph over the same weights
        double[][] loopOnce = Reference(looped: true, passes: 1).Logits(Prompt);
        double[][] plain = Reference(looped: false).Logits(Prompt);
        Assert.True(MaxAbsDiff(loopOnce, plain) <= 1e-12);
    }

    [Fact]
    public void EveryMechanismIsExercised()
    {
        var reference = Reference(looped: true);
        double cap = Builder(looped: true).Softcap;
        var c = new Inkling2Reference.Counters();
        reference.Logits(Prompt, c);
        _output.WriteLine($"clipped rows: q {c.QRowsClipped}/{c.QRows}, k {c.KRowsClipped}/{c.KRows}, r {c.RRowsClipped}/{c.RRows}");
        _output.WriteLine($"global pairs past the extent {c.GlobalPairsPastExtent}/{c.GlobalPairs}, sliding pairs masked {c.SlidingPairsMasked}/{c.SlidingPairs}");
        _output.WriteLine($"max tau {c.MaxTau:F3}, sink mass mean {c.SinkMassSum / c.SinkMassCount:F3} max {c.SinkMassMax:F3}, max |t| {c.MaxAbsSink:F2}");
        _output.WriteLine($"routings changed by exp_probs_b {c.RoutingsChangedByBias}/{c.Routings}");
        _output.WriteLine($"SITU gate past beta {c.SituGatePastBeta}/{c.SituValues}, up past linear beta {c.SituUpPastLinearBeta}/{c.SituValues}");
        _output.WriteLine($"softcap: max |z| / cap {c.MaxAbsPreSoftcap / cap:F2}, mean {c.PreSoftcapRatioSum / c.LogitCount:F2}");
        _output.WriteLine($"message layer-tokens {c.MessageLayerTokens}");

        static void Sometimes(long hits, long total, string what)
            => Assert.True(hits > 0 && hits < total, $"{what}: {hits} of {total}");
        Sometimes(c.QRowsClipped, c.QRows, "clipped q rows");
        Sometimes(c.KRowsClipped, c.KRows, "clipped k rows");
        Sometimes(c.RRowsClipped, c.RRows, "clipped r rows");
        Sometimes(c.GlobalPairsPastExtent, c.GlobalPairs, "global pairs on the zero bias column");
        Sometimes(c.SlidingPairsMasked, c.SlidingPairs, "sliding pairs outside the window");
        Sometimes(c.RoutingsChangedByBias, c.Routings, "routings exp_probs_b changed");
        Sometimes(c.SituGatePastBeta, c.SituValues, "SITU gates past beta");
        Sometimes(c.SituUpPastLinearBeta, c.SituValues, "SITU ups past the linear beta");
        Assert.True(c.MaxTau > 1.2, $"max tau {c.MaxTau}");
        double sinkMean = c.SinkMassSum / c.SinkMassCount;
        Assert.True(sinkMean > 0.01 && sinkMean < 0.9, $"mean sink mass {sinkMean}");
        double capRatio = c.MaxAbsPreSoftcap / cap;
        Assert.True(capRatio > 0.5 && capRatio < 3, $"max |z| / cap {capRatio}");
        // passes 1 and 2 over the four core layers read messages, every token
        Assert.Equal(2 * 4 * Prompt.Length, c.MessageLayerTokens);
    }
}
