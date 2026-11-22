// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

/// <summary>
/// An independent Inkling 2 forward in double precision over an F32 GGUF (<c>inkling2</c> or <c>inkling2_loop</c>), written
/// from llama.cpp's graph (src/models/inkling2.cpp, inkling2-loop.cpp, llama-graph.cpp's build_attn_mha and build_moe_ffn,
/// llama-kv-cache.cpp's set_input_relative_pos) and nothing of TensorSharp's but the GGUF file reader. Everything it needs
/// comes from the file: the hyperparameters from the metadata, the weights from the F32 tensors, each checked against the
/// shape llama.cpp's loader creates.
/// <para>
/// One sequence, computed chunk by chunk as llama.cpp's graph does a ubatch: every token of the chunk through a layer, then
/// the next layer. The state carried between chunks is exactly what llama.cpp keeps in its memory, per cache slot (a
/// physical layer, or a core layer's later pass at slot <c>n_phys + (pass - 1) * core + (ip - start)</c>): the K/V rows
/// of every earlier position and the last <c>taps - 1</c> inputs of the four convolutions (K, V, the attention output, the
/// MoE output). The block residuals, the loop bank and the messages are per token and never cross a chunk.
/// <see cref="Logits(int[], Counters)"/> runs the whole sequence as one chunk from an empty state (zero conv history,
/// no keys); <see cref="LogitsChunked"/> splits it.
/// </para>
/// </summary>
internal sealed class Inkling2Reference
{
    /// <summary>The hyperparameters, as llama.cpp's load_arch_hparams reads them.</summary>
    public sealed class Config
    {
        public string Architecture;
        public int Layers, Hidden, HeadDim, Experts, ExpertsUsed, ExpertFf, Vocab, VocabPadded;
        public int[] Heads, KvHeads;
        public bool[] Sliding;
        public int Window, Taps, BlockSize, RelativeLength, RelativeExtent;
        public double Eps, ExpertWeightsScale, SituBeta, SituLinearBeta, Softcap, QkClip, RelativeClip, LogScaleFloor, LogScaleAlpha;
        public bool Looped;
        public int LoopStart, LoopEnd, LoopCount, GateRank;
        public bool LoopVarying;

        /// <summary>attn_rel_bias columns of layer <paramref name="l"/>: the window on sliding layers.</summary>
        public int Extent(int l) => Sliding[l] ? Window : RelativeExtent;
        public int Slots => Layers + (Looped ? (LoopCount - 1) * (LoopEnd - LoopStart) : 0);
    }

    /// <summary>One layer application: its cache slot, the physical layer whose weights it reads, and its loop pass (-1
    /// outside the loop core).</summary>
    public readonly record struct VirtualLayer(int Slot, int Physical, int Pass);

    /// <summary>What one cache slot carries between chunks.</summary>
    public sealed class SlotState
    {
        /// <summary>Per position: K after its conv and clip, V after its conv and per-head RMS norm, [kvHeads * headDim].</summary>
        public readonly List<double[]> Keys = new(), Values = new();
        /// <summary>The last taps - 1 conv inputs, oldest first: [taps - 1][channels].</summary>
        public double[][] ConvK, ConvV, ConvAttn, ConvFfn;
    }

    /// <summary>The sequence's memory: its next position and every slot's state.</summary>
    public sealed class State
    {
        public int Position;
        public SlotState[] Slots;
    }

    /// <summary>How often each mechanism actually did something (tests check none is idle or saturated).</summary>
    public sealed class Counters
    {
        public long QRows, QRowsClipped, KRows, KRowsClipped, RRows, RRowsClipped;
        /// <summary>Causal (query, key) pairs of global layers, and those at distance &gt;= extent (the zero bias column).</summary>
        public long GlobalPairs, GlobalPairsPastExtent;
        /// <summary>Causal pairs of sliding layers, and those outside the window (masked).</summary>
        public long SlidingPairs, SlidingPairsMasked;
        public double MaxTau = 1;
        /// <summary>Softmax mass on the zero sink logit, over every (query, head).</summary>
        public double SinkMassSum, SinkMassMax;
        public long SinkMassCount;
        public double MaxAbsSink;
        /// <summary>Token routings, and those whose expert set exp_probs_b changed.</summary>
        public long Routings, RoutingsChangedByBias;
        /// <summary>SITU inputs: gate values past beta and up values past the linear beta (where tanh bends).</summary>
        public long SituValues, SituGatePastBeta, SituUpPastLinearBeta;
        /// <summary>Layer applications that read cross-pass messages.</summary>
        public long MessageLayerTokens;
        /// <summary>Output logits before the softcap: the largest |z| and the sum of |z| / cap.</summary>
        public double MaxAbsPreSoftcap, PreSoftcapRatioSum;
        public long LogitCount;
        /// <summary>Set (not null), every token routing: its position, layer run, chosen experts and selection scores.</summary>
        public List<Routing> RoutingLog;
    }

    /// <summary>One token's routing in one layer run (the runtime's layer index: <see cref="Schedule"/>'s order): the chosen
    /// experts and the scores they were picked by (sigmoid probability plus exp_probs_b, per expert).</summary>
    public sealed record Routing(int Position, int Run, int[] Chosen, double[] Scores);

    /// <summary>Set, a routing takes the experts this gives for (position, layer run) instead of its top-k (null: the top-k);
    /// the weights are still the chosen experts' normalized probabilities. Tests run the reference along the routes a
    /// device picked.</summary>
    public Func<int, int, int[]> RouteOverride { get; set; }

    private static readonly string[] CtrlNames = { "q", "k", "v", "ffn" };
    private const double MixEps = 1e-6;   // ggml_rms_norm(..., 1e-6f) in mix() and control()

    private readonly Dictionary<string, float[]> _w;
    public Config Cfg { get; }
    public IReadOnlyList<VirtualLayer> Schedule { get; }

    private Inkling2Reference(Config cfg, Dictionary<string, float[]> weights, Dictionary<string, ulong[]> shapes)
    {
        Cfg = cfg;
        _w = weights;
        CheckShapes(shapes);
        Schedule = BuildSchedule();
    }

    /// <summary>Read the hyperparameters and the F32 weights of an Inkling 2 GGUF.</summary>
    public static Inkling2Reference Load(string path)
    {
        using var file = new GgufFile(path);
        string a = file.GetString("general.architecture");
        if (a != "inkling2" && a != "inkling2_loop")
            throw new InvalidDataException($"'{path}' is a {a} GGUF, not inkling2 or inkling2_loop.");
        uint U(string key) => file.Metadata.ContainsKey($"{a}.{key}") ? file.GetUint32($"{a}.{key}")
            : throw new InvalidDataException($"missing {a}.{key}");
        double F(string key) => file.Metadata.ContainsKey($"{a}.{key}") ? file.GetFloat32($"{a}.{key}")
            : throw new InvalidDataException($"missing {a}.{key}");
        int layers = (int)U("block_count");
        int[] PerLayer(string key)
        {
            int[] values = file.GetInt32Array($"{a}.{key}");
            if (values == null) return Enumerable.Repeat((int)U(key), layers).ToArray();
            if (values.Length != layers) throw new InvalidDataException($"{a}.{key} has {values.Length} entries for {layers} layers");
            return values;
        }
        var cfg = new Config
        {
            Architecture = a,
            Layers = layers,
            Hidden = (int)U("embedding_length"),
            HeadDim = (int)U("attention.key_length"),
            Heads = PerLayer("attention.head_count"),
            KvHeads = PerLayer("attention.head_count_kv"),
            Eps = F("attention.layer_norm_rms_epsilon"),
            Experts = (int)U("expert_count"),
            ExpertsUsed = (int)U("expert_used_count"),
            ExpertFf = (int)U("expert_feed_forward_length"),
            Vocab = (int)U("vocab_size_unpadded"),
            VocabPadded = (int)U("vocab_size"),
            Window = (int)U("attention.sliding_window"),
            Sliding = file.GetBoolArray($"{a}.attention.sliding_window_pattern") ?? throw new InvalidDataException("missing sliding_window_pattern"),
            Taps = (int)U("shortconv.l_cache"),
            ExpertWeightsScale = file.GetFloat32($"{a}.expert_weights_scale", 0f),
            SituBeta = F("activation.situ_beta"),
            SituLinearBeta = F("activation.situ_linear_beta"),
            BlockSize = (int)U("attn_res.block_size"),
            Softcap = file.GetFloat32($"{a}.final_logit_softcapping", 0f),
            RelativeLength = (int)U("attention.relative_length"),
            RelativeExtent = (int)U("attention.relative_extent"),
            QkClip = F("attention.qk_clip"),
            RelativeClip = F("attention.relative_clip"),
            LogScaleFloor = U("attention.log_scale_floor"),
            LogScaleAlpha = F("attention.log_scale_alpha"),
            Looped = a == "inkling2_loop",
        };
        if ((int)U("attention.value_length") != cfg.HeadDim)
            throw new InvalidDataException("inkling2 needs key_length == value_length");
        if (U("rope.dimension_count") != 0)
            throw new InvalidDataException("inkling2 has no RoPE (rope.dimension_count 0)");
        if (cfg.Sliding.Length != layers || cfg.Vocab > cfg.VocabPadded || cfg.Taps < 2 || cfg.BlockSize < 1 || cfg.LogScaleFloor <= 0)
            throw new InvalidDataException("inkling2: invalid dimensions, vocabulary, or layer pattern");
        if (cfg.Looped)
        {
            cfg.LoopStart = (int)U("loop.start");
            cfg.LoopEnd = (int)U("loop.end");
            cfg.LoopCount = (int)U("loop.count");
            cfg.GateRank = (int)U("loop.gate_rank");
            cfg.LoopVarying = file.GetBool($"{a}.loop.varying", false);
            if (cfg.LoopCount < (cfg.LoopVarying ? 1 : 2) || cfg.GateRank < 1 || cfg.LoopStart <= 0 || cfg.LoopStart >= cfg.LoopEnd ||
                cfg.LoopEnd > layers || cfg.LoopStart % cfg.BlockSize != 0 || cfg.LoopEnd % cfg.BlockSize != 0)
                throw new InvalidDataException($"inkling2_loop: invalid loop core [{cfg.LoopStart}, {cfg.LoopEnd}) or {cfg.LoopCount} passes");
        }

        var weights = new Dictionary<string, float[]>();
        var shapes = new Dictionary<string, ulong[]>();
        foreach (GgufTensorInfo info in file.Tensors.Values)
        {
            if (info.Type != GgmlTensorType.F32 && info.Type != GgmlTensorType.F16)
                throw new InvalidDataException($"the reference reads F32 and F16 only: '{info.Name}' is {info.Type}");
            byte[] raw = file.ReadTensorData(info);
            var values = new float[info.NumElements];
            if (info.Type == GgmlTensorType.F32)
                Buffer.BlockCopy(raw, 0, values, 0, raw.Length);
            else
                for (long i = 0; i < values.LongLength; i++)
                    values[i] = (float)BitConverter.ToHalf(raw, checked((int)(2 * i)));
            weights[info.Name] = values;
            shapes[info.Name] = info.Shape;
        }
        return new Inkling2Reference(cfg, weights, shapes);
    }

    /// <summary>The raw F32 values of a tensor (ne0 fastest).</summary>
    public float[] Weight(string name) => W(name);

    private float[] W(string name) => _w.TryGetValue(name, out float[] v) ? v : throw new KeyNotFoundException($"missing tensor '{name}'");

    /// <summary>Every tensor llama.cpp's loader creates, with its shape, and nothing else in the file.</summary>
    private void CheckShapes(Dictionary<string, ulong[]> shapes)
    {
        var c = Cfg;
        var expected = new Dictionary<string, long[]>
        {
            ["token_embd.weight"] = new long[] { c.Hidden, c.VocabPadded },
            ["token_embd_norm.weight"] = new long[] { c.Hidden },
            ["output.weight"] = new long[] { c.Hidden, c.VocabPadded },
            ["output_norm.weight"] = new long[] { c.Hidden },
            ["output_res_query.weight"] = new long[] { c.Hidden },
        };
        for (int l = 0; l < c.Layers; l++)
        {
            string p = $"blk.{l}.";
            long H = c.Hidden, qd = c.Heads[l] * c.HeadDim, kvd = c.KvHeads[l] * c.HeadDim, taps = c.Taps, rel = c.RelativeLength;
            if (c.KvHeads[l] <= 0 || c.Heads[l] % c.KvHeads[l] != 0 || c.ExpertsUsed > c.Experts)
                throw new InvalidDataException($"inkling2: invalid attention or expert dimensions at layer {l}");
            expected[p + "attn_q.weight"] = new[] { H, qd };
            expected[p + "attn_k.weight"] = new[] { H, kvd };
            expected[p + "attn_v.weight"] = new[] { H, kvd };
            expected[p + "attn_output.weight"] = new[] { qd, H };
            expected[p + "attn_gate.weight"] = new[] { H, qd };
            expected[p + "attn_q_clip.weight"] = new long[] { c.HeadDim, c.Heads[l] };
            expected[p + "attn_sink_proj.weight"] = new long[] { H, c.Heads[l] };
            expected[p + "attn_rel.weight"] = new[] { H, rel * c.Heads[l] };
            expected[p + "attn_rel_bias.weight"] = new long[] { rel, c.Extent(l) };
            expected[p + "attn_rel_clip.weight"] = new long[] { rel, c.Heads[l] };
            expected[p + "attn_k_conv.weight"] = new[] { taps, kvd };
            expected[p + "attn_v_conv.weight"] = new[] { taps, kvd };
            expected[p + "attn_post_conv.weight"] = new[] { taps, H };
            expected[p + "ffn_post_conv.weight"] = new[] { taps, H };
            if (c.Sliding[l])
            {
                expected[p + "attn_norm.weight"] = new[] { H };
            }
            else
            {
                expected[p + "attn_q_inp_norm.weight"] = new[] { H };
                expected[p + "attn_k_inp_norm.weight"] = new[] { H };
                expected[p + "attn_v_inp_norm.weight"] = new[] { H };
                expected[p + "attn_k_conv_proj.weight"] = new[] { H, taps * c.KvHeads[l] };
                expected[p + "attn_v_conv_proj.weight"] = new[] { H, taps * c.KvHeads[l] };
            }
            // [n_embd, 1] is stored as [n_embd]: ggml treats trailing ones as absent.
            if (l > 0) expected[p + "attn_res_query.weight"] = c.Sliding[l] ? new[] { H } : new[] { H, 3 };
            expected[p + "ffn_res_query.weight"] = new[] { H };
            expected[p + "ffn_norm.weight"] = new[] { H };
            expected[p + "ffn_gate_inp.weight"] = new long[] { H, c.Experts };
            expected[p + "exp_probs_b.bias"] = new long[] { c.Experts };
            expected[p + "ffn_gate_up_exps.weight"] = new long[] { H, 2 * c.ExpertFf, c.Experts };
            expected[p + "ffn_down_exps.weight"] = new long[] { c.ExpertFf, H, c.Experts };
            if (c.Looped && l >= c.LoopStart && l < c.LoopEnd)
            {
                expected[p + "loop_msg_query.weight"] = new[] { H, 4 };
                foreach (string n in CtrlNames)
                {
                    expected[p + $"loop_{n}_ctrl_down.weight"] = new[] { 2 * H, c.GateRank };
                    expected[p + $"loop_{n}_ctrl_gain.weight"] = new[] { c.GateRank, H };
                    expected[p + $"loop_{n}_ctrl_forget.weight"] = new[] { c.GateRank, H };
                }
            }
        }
        // ggml shapes match up to trailing ones ({n_embd, 1} may be written as [n_embd])
        static IEnumerable<long> Trim(IEnumerable<long> dims) => dims.Reverse().SkipWhile(x => x == 1).Reverse();
        foreach (var (name, dims) in expected)
        {
            if (!shapes.TryGetValue(name, out ulong[] actual))
                throw new InvalidDataException($"missing tensor '{name}'");
            if (!Trim(actual.Select(x => (long)x)).SequenceEqual(Trim(dims)))
                throw new InvalidDataException($"'{name}' is [{string.Join(", ", actual)}], expected [{string.Join(", ", dims)}]");
        }
        string extra = shapes.Keys.FirstOrDefault(n => !expected.ContainsKey(n));
        if (extra != null)
            throw new InvalidDataException($"unexpected tensor '{extra}'");
    }

    private List<VirtualLayer> BuildSchedule()
    {
        var c = Cfg;
        var s = new List<VirtualLayer>();
        if (!c.Looped)
        {
            for (int l = 0; l < c.Layers; l++) s.Add(new VirtualLayer(l, l, -1));
            return s;
        }
        for (int l = 0; l < c.LoopStart; l++) s.Add(new VirtualLayer(l, l, -1));
        for (int pass = 0; pass < c.LoopCount; pass++)
            for (int ip = c.LoopStart; ip < c.LoopEnd; ip++)
                s.Add(new VirtualLayer(Slot(pass, ip), ip, pass));
        for (int l = c.LoopEnd; l < c.Layers; l++) s.Add(new VirtualLayer(l, l, -1));
        return s;
    }

    /// <summary>llama.cpp's cache slot of core layer <paramref name="ip"/> in pass <paramref name="pass"/>.</summary>
    private int Slot(int pass, int ip) => pass == 0 ? ip : Cfg.Layers + (pass - 1) * (Cfg.LoopEnd - Cfg.LoopStart) + (ip - Cfg.LoopStart);

    /// <summary>An empty sequence: position 0, no keys, zero conv history.</summary>
    public State NewState()
    {
        var slots = new SlotState[Cfg.Slots];
        for (int s = 0; s < slots.Length; s++)
        {
            int l = PhysicalOfSlot(s);
            int kvd = Cfg.KvHeads[l] * Cfg.HeadDim;
            slots[s] = new SlotState
            {
                ConvK = Zeros(Cfg.Taps - 1, kvd),
                ConvV = Zeros(Cfg.Taps - 1, kvd),
                ConvAttn = Zeros(Cfg.Taps - 1, Cfg.Hidden),
                ConvFfn = Zeros(Cfg.Taps - 1, Cfg.Hidden),
            };
        }
        return new State { Position = 0, Slots = slots };
    }

    private int PhysicalOfSlot(int slot)
        => slot < Cfg.Layers ? slot : Cfg.LoopStart + (slot - Cfg.Layers) % (Cfg.LoopEnd - Cfg.LoopStart);

    /// <summary>The whole sequence in one pass: logits [position][vocab_size_unpadded], after the softcap.</summary>
    public double[][] Logits(int[] tokens, Counters counters = null) => Forward(NewState(), tokens, counters);

    /// <summary>The sequence split into chunks of <paramref name="chunkSizes"/> (which must sum to its length), the state
    /// carried from one chunk to the next.</summary>
    public double[][] LogitsChunked(int[] tokens, IEnumerable<int> chunkSizes, Counters counters = null)
    {
        State state = NewState();
        var rows = new List<double[]>();
        int at = 0;
        foreach (int n in chunkSizes)
        {
            rows.AddRange(Forward(state, tokens[at..(at + n)], counters));
            at += n;
        }
        if (at != tokens.Length) throw new ArgumentException($"chunks cover {at} of {tokens.Length} tokens");
        return rows.ToArray();
    }

    /// <summary>One chunk at the state's position; advances the state.</summary>
    public double[][] Forward(State state, int[] tokens, Counters counters = null)
    {
        var c = Cfg;
        counters ??= new Counters();
        int T = tokens.Length, p0 = state.Position;
        float[] embd = W("token_embd.weight");
        var cur = new double[T][];
        for (int t = 0; t < T; t++)
        {
            if (tokens[t] < 0 || tokens[t] >= c.VocabPadded) throw new ArgumentOutOfRangeException(nameof(tokens));
            var row = new double[c.Hidden];
            for (int i = 0; i < c.Hidden; i++) row[i] = embd[tokens[t] * c.Hidden + i];
            cur[t] = RmsNorm(row, W("token_embd_norm.weight"), c.Eps);
        }

        var blocks = new List<double[][]>();
        int run = 0;   // the layer run: the runtime's layer index
        if (!c.Looped)
        {
            for (int l = 0; l < c.Layers; l++) cur = Layer(state.Slots[l], l, cur, blocks, null, null, p0, counters, run++);
        }
        else
        {
            for (int l = 0; l < c.LoopStart; l++) cur = Layer(state.Slots[l], l, cur, blocks, null, null, p0, counters, run++);
            // every pass starts from the prelude output and blocks
            double[][] prelude = cur;
            var preludeBlocks = new List<double[][]>(blocks);
            for (int pass = 0; pass < c.LoopCount; pass++)
            {
                List<double[][]> bank = null;
                double[][] bankOut = null;
                if (pass > 0)
                {
                    // the previous pass's blocks after its first core block, and its output
                    bank = blocks.Skip(preludeBlocks.Count + 1).ToList();
                    bankOut = cur;
                    cur = prelude;
                    blocks = new List<double[][]>(preludeBlocks);
                }
                for (int ip = c.LoopStart; ip < c.LoopEnd; ip++)
                    cur = Layer(state.Slots[Slot(pass, ip)], ip, cur, blocks, bank, bankOut, p0, counters, run++);
            }
            for (int l = c.LoopEnd; l < c.Layers; l++) cur = Layer(state.Slots[l], l, cur, blocks, null, null, p0, counters, run++);
        }

        // output: mix, norm, lm_head, softcap, drop the padded vocabulary
        float[] head = W("output.weight");
        float[] outQuery = W("output_res_query.weight");
        var logits = new double[T][];
        for (int t = 0; t < T; t++)
        {
            double[] x = RmsNorm(Mix(BlocksAt(blocks, t), cur[t], outQuery, 1)[0], W("output_norm.weight"), c.Eps);
            var row = new double[c.Vocab];
            for (int v = 0; v < c.Vocab; v++)
            {
                double z = Dot(head, v * c.Hidden, x);
                if (c.Softcap != 0)
                {
                    counters.MaxAbsPreSoftcap = Math.Max(counters.MaxAbsPreSoftcap, Math.Abs(z));
                    counters.PreSoftcapRatioSum += Math.Abs(z) / c.Softcap;
                    z = c.Softcap * Math.Tanh(z / c.Softcap);
                }
                counters.LogitCount++;
                row[v] = z;
            }
            logits[t] = row;
        }
        state.Position += T;
        return logits;
    }

    /// <summary>Physical layer <paramref name="l"/> over the chunk (llama.cpp's build_layer with ip = l), its state in
    /// <paramref name="st"/>; with a bank, a later loop pass that reads messages.</summary>
    private double[][] Layer(SlotState st, int l, double[][] cur, List<double[][]> blocks, List<double[][]> bank, double[][] bankOut,
        int p0, Counters counters, int run)
    {
        var c = Cfg;
        string p = $"blk.{l}.";
        bool swa = c.Sliding[l];
        int T = cur.Length, H = c.Hidden, d = c.HeadDim, nh = c.Heads[l], nkv = c.KvHeads[l];
        int qd = nh * d, kvd = nkv * d, rel = c.RelativeLength, taps = c.Taps, group = nh / nkv;

        // the messages of a later pass: mix the previous pass's bank with this layer's q, k, v and ffn queries
        double[][][] messages = null;
        if (bank != null)
        {
            float[] mq = W(p + "loop_msg_query.weight");
            messages = new double[T][][];
            for (int t = 0; t < T; t++) messages[t] = Mix(BlocksAt(bank, t), bankOut[t], mq, 4);
            counters.MessageLayerTokens += T;
        }

        double[][] inpSA = cur;

        // attention inputs: sliding layers mix one, global layers one each for Q, K and V
        var inputs = new double[3][][];
        for (int j = 0; j < 3; j++) inputs[j] = new double[T][];
        for (int t = 0; t < T; t++)
        {
            if (l > 0)
            {
                double[][] mixed = Mix(BlocksAt(blocks, t), cur[t], W(p + "attn_res_query.weight"), swa ? 1 : 3);
                for (int j = 0; j < 3; j++) inputs[j][t] = mixed[swa ? 0 : j];
            }
            else
            {
                for (int j = 0; j < 3; j++) inputs[j][t] = cur[t];
            }
            if (messages != null)
                for (int j = 0; j < 3; j++) inputs[j][t] = Control(inputs[j][t], messages[t][j], p, j);
            // one norm on sliding layers (applied per input: equal inputs stay equal), three on global ones
            for (int j = 0; j < 3; j++)
            {
                string norm = swa ? "attn_norm" : $"attn_{"qkv"[j]}_inp_norm";
                inputs[j][t] = RmsNorm(inputs[j][t], W(p + norm + ".weight"), c.Eps);
            }
        }

        // projections
        var kRaw = new double[T][];
        var vRaw = new double[T][];
        var dynK = swa ? null : new double[T][];
        var dynV = swa ? null : new double[T][];
        var q = new double[T][];
        var r = new double[T][];
        var sinks = new double[T][];
        var gates = new double[T][];
        for (int t = 0; t < T; t++)
        {
            double[] xq = inputs[0][t], xk = inputs[1][t], xv = inputs[2][t];
            q[t] = MatVec(W(p + "attn_q.weight"), 0, xq, qd);
            kRaw[t] = MatVec(W(p + "attn_k.weight"), 0, xk, kvd);
            vRaw[t] = MatVec(W(p + "attn_v.weight"), 0, xv, kvd);
            r[t] = MatVec(W(p + "attn_rel.weight"), 0, xq, rel * nh);
            sinks[t] = MatVec(W(p + "attn_sink_proj.weight"), 0, xq, nh);
            gates[t] = MatVec(W(p + "attn_gate.weight"), 0, xq, qd);
            if (!swa)
            {
                dynK[t] = MatVec(W(p + "attn_k_conv_proj.weight"), 0, xk, taps * nkv);
                dynV[t] = MatVec(W(p + "attn_v_conv_proj.weight"), 0, xv, taps * nkv);
            }
        }
        double[][] k = Convolve(kRaw, ref st.ConvK, W(p + "attn_k_conv.weight"), dynK, d);
        double[][] v = Convolve(vRaw, ref st.ConvV, W(p + "attn_v_conv.weight"), dynV, d);

        float[] qClip = W(p + "attn_q_clip.weight"), relClip = W(p + "attn_rel_clip.weight");
        for (int t = 0; t < T; t++)
        {
            for (int h = 0; h < nh; h++)
            {
                counters.QRows++;
                if (Clip(q[t], h * d, d, qClip, c.QkClip)) counters.QRowsClipped++;
                counters.RRows++;
                if (Clip(r[t], h * rel, rel, relClip, c.RelativeClip)) counters.RRowsClipped++;
            }
            for (int h = 0; h < nkv; h++)
            {
                counters.KRows++;
                if (Clip(k[t], h * d, d, null, c.QkClip)) counters.KRowsClipped++;
                HeadRmsNorm(v[t], h * d, d, c.Eps);
            }
            if (!swa)
            {
                // log-scaled positions: 1 + alpha * log(max(1, (pos + 1) / floor))
                double tau = 1 + c.LogScaleAlpha * Math.Log(Math.Max(1.0, (p0 + t + 1) / c.LogScaleFloor));
                counters.MaxTau = Math.Max(counters.MaxTau, tau);
                for (int i = 0; i < qd; i++) q[t][i] *= tau;
                for (int i = 0; i < rel * nh; i++) r[t][i] *= tau;
            }
            st.Keys.Add(k[t]);
            st.Values.Add(v[t]);
        }

        // attention: s_ij = q.k / sqrt(d) + r.B[:, d_ij] - t_i, one extra zero logit, causal (and windowed on sliding layers)
        float[] bias = W(p + "attn_rel_bias.weight");
        int extent = c.Extent(l);
        double scale = 1.0 / Math.Sqrt(d);
        var preOut = new double[T][];
        for (int t = 0; t < T; t++)
        {
            int pos = p0 + t;
            var attn = new double[qd];
            for (int h = 0; h < nh; h++)
            {
                int kvh = h / group;
                counters.MaxAbsSink = Math.Max(counters.MaxAbsSink, Math.Abs(sinks[t][h]));
                // the bias table of this query and head: r . B[:, e] for e < extent, the padded zero column at extent
                var table = new double[extent + 1];
                for (int e = 0; e < extent; e++)
                {
                    double s = 0;
                    for (int i = 0; i < rel; i++) s += r[t][h * rel + i] * bias[e * rel + i];
                    table[e] = s;
                }
                var scores = new double[pos + 1];
                var visible = new bool[pos + 1];
                double max = 0;   // the zero sink logit
                for (int j = 0; j <= pos; j++)
                {
                    int distance = pos - j;
                    if (swa)
                    {
                        if (h == 0) counters.SlidingPairs++;
                        if (distance >= c.Window)
                        {
                            if (h == 0) counters.SlidingPairsMasked++;
                            continue;
                        }
                    }
                    else if (h == 0)
                    {
                        counters.GlobalPairs++;
                        if (distance >= extent) counters.GlobalPairsPastExtent++;
                    }
                    int column = distance >= 0 && distance < extent ? distance : extent;
                    double dot = 0;
                    double[] key = st.Keys[j];
                    for (int i = 0; i < d; i++) dot += q[t][h * d + i] * key[kvh * d + i];
                    scores[j] = dot * scale + table[column] - sinks[t][h];
                    visible[j] = true;
                    max = Math.Max(max, scores[j]);
                }
                double sum = Math.Exp(-max);   // exp(0 - max): the sink
                for (int j = 0; j <= pos; j++)
                    if (visible[j]) sum += scores[j] = Math.Exp(scores[j] - max);
                double sinkMass = Math.Exp(-max) / sum;
                counters.SinkMassSum += sinkMass;
                counters.SinkMassMax = Math.Max(counters.SinkMassMax, sinkMass);
                counters.SinkMassCount++;
                for (int j = 0; j <= pos; j++)
                {
                    if (!visible[j]) continue;
                    double w = scores[j] / sum;
                    double[] value = st.Values[j];
                    for (int i = 0; i < d; i++) attn[h * d + i] += w * value[kvh * d + i];
                }
            }
            for (int i = 0; i < qd; i++) attn[i] *= Sigmoid(gates[t][i]);
            preOut[t] = MatVec(W(p + "attn_output.weight"), 0, attn, H);
        }
        double[][] o = Convolve(preOut, ref st.ConvAttn, W(p + "attn_post_conv.weight"), null, 1);

        // the first layer of each block starts a new residual stream, the earlier one joins the mixed blocks
        double[][] ffnInp;
        if (l % c.BlockSize == 0)
        {
            blocks.Add(inpSA);
            ffnInp = o;
        }
        else
        {
            ffnInp = new double[T][];
            for (int t = 0; t < T; t++) ffnInp[t] = Add(inpSA[t], o[t]);
        }

        // MoE
        var moe = new double[T][];
        float[] ffnQuery = W(p + "ffn_res_query.weight");
        for (int t = 0; t < T; t++)
        {
            double[] x = Mix(BlocksAt(blocks, t), ffnInp[t], ffnQuery, 1)[0];
            if (messages != null) x = Control(x, messages[t][3], p, 3);
            x = RmsNorm(x, W(p + "ffn_norm.weight"), c.Eps);
            moe[t] = Moe(p, x, counters, p0 + t, run);
        }
        double[][] ffnOut = Convolve(moe, ref st.ConvFfn, W(p + "ffn_post_conv.weight"), null, 1);
        var result = new double[T][];
        for (int t = 0; t < T; t++) result[t] = Add(ffnInp[t], ffnOut[t]);
        return result;
    }

    /// <summary>
    /// Sigmoid routing (F32 router in llama.cpp), top-k by p + exp_probs_b (the bias selects only), weights
    /// p / (sum + 1e-20) * expert_weights_scale, SITU on the fused gate_up (gate = the first ff rows), down, weighted sum.
    /// </summary>
    private double[] Moe(string p, double[] x, Counters counters, int position, int run)
    {
        var c = Cfg;
        int H = c.Hidden, E = c.Experts, ff = c.ExpertFf, k = c.ExpertsUsed;
        double[] logits = MatVec(W(p + "ffn_gate_inp.weight"), 0, x, E);
        float[] b = W(p + "exp_probs_b.bias");
        var probs = new double[E];
        var selection = new double[E];
        for (int e = 0; e < E; e++)
        {
            probs[e] = Sigmoid(logits[e]);
            selection[e] = probs[e] + b[e];
        }
        int[] chosen = RouteOverride?.Invoke(position, run) ?? TopK(selection, k);
        if (chosen.Length != k || chosen.Distinct().Count() != k || chosen.Any(e => e < 0 || e >= E))
            throw new ArgumentException($"routing at position {position}, run {run}: {k} distinct experts below {E} expected");
        counters.RoutingLog?.Add(new Routing(position, run, chosen, selection));
        int[] unbiased = TopK(probs, k);
        counters.Routings++;
        if (!chosen.OrderBy(e => e).SequenceEqual(unbiased.OrderBy(e => e))) counters.RoutingsChangedByBias++;

        double sum = 0;
        foreach (int e in chosen) sum += probs[e];
        double weightScale = c.ExpertWeightsScale != 0 && c.ExpertWeightsScale != 1 ? c.ExpertWeightsScale : 1;
        float[] gateUp = W(p + "ffn_gate_up_exps.weight"), down = W(p + "ffn_down_exps.weight");
        var y = new double[H];
        foreach (int e in chosen)
        {
            double weight = probs[e] / (sum + 1e-20) * weightScale;
            double[] gu = MatVec(gateUp, e * 2 * ff * H, x, 2 * ff);
            var act = new double[ff];
            for (int i = 0; i < ff; i++)
            {
                double g = gu[i], u = gu[ff + i];
                counters.SituValues++;
                if (Math.Abs(g) > c.SituBeta) counters.SituGatePastBeta++;
                if (Math.Abs(u) > c.SituLinearBeta) counters.SituUpPastLinearBeta++;
                double gateAct = c.SituBeta * Math.Tanh(g / c.SituBeta) * Sigmoid(g);
                double upAct = c.SituLinearBeta > 0 ? c.SituLinearBeta * Math.Tanh(u / c.SituLinearBeta) : u;
                act[i] = gateAct * upAct;
            }
            double[] o = MatVec(down, e * H * ff, act, H);
            for (int i = 0; i < H; i++) y[i] += weight * o[i];
        }
        return y;
    }

    /// <summary>The <paramref name="k"/> largest entries' indices, largest first (ties: the lower index).</summary>
    private static int[] TopK(double[] values, int k)
        => Enumerable.Range(0, values.Length).OrderByDescending(i => values[i]).ThenBy(i => i).Take(k).ToArray();

    /// <summary>
    /// mix(blocks, cur, query): the states (the blocks and cur) RMS-normalized without weight, scores S . q / sqrt(n_embd),
    /// a softmax over the states, and their weighted sum; one output per query (query j at j * n_embd).
    /// </summary>
    private double[][] Mix(IEnumerable<double[]> blocks, double[] cur, float[] query, int queries)
    {
        int H = Cfg.Hidden;
        var states = blocks.Append(cur).Select(s => RmsNorm(s, null, MixEps)).ToList();
        var outputs = new double[queries][];
        double scale = 1.0 / Math.Sqrt(H);
        for (int j = 0; j < queries; j++)
        {
            var scores = new double[states.Count];
            double max = double.NegativeInfinity;
            for (int s = 0; s < states.Count; s++)
            {
                scores[s] = Dot(query, j * H, states[s]) * scale;
                max = Math.Max(max, scores[s]);
            }
            double sum = 0;
            for (int s = 0; s < states.Count; s++) sum += scores[s] = Math.Exp(scores[s] - max);
            var o = new double[H];
            for (int s = 0; s < states.Count; s++)
                for (int i = 0; i < H; i++) o[i] += scores[s] / sum * states[s][i];
            outputs[j] = o;
        }
        return outputs;
    }

    /// <summary>control(x, m): g = W_down . [rms(x); rms(m)], x - tanh(W_forget . g) * x + tanh(W_gain . g) * m.</summary>
    private double[] Control(double[] x, double[] m, string p, int ctrl)
    {
        int H = Cfg.Hidden;
        string n = CtrlNames[ctrl];
        double[] z = RmsNorm(x, null, MixEps).Concat(RmsNorm(m, null, MixEps)).ToArray();
        double[] g = MatVec(W(p + $"loop_{n}_ctrl_down.weight"), 0, z, Cfg.GateRank);
        double[] gain = MatVec(W(p + $"loop_{n}_ctrl_gain.weight"), 0, g, H);
        double[] forget = MatVec(W(p + $"loop_{n}_ctrl_forget.weight"), 0, g, H);
        var y = new double[H];
        for (int i = 0; i < H; i++) y[i] = x[i] - Math.Tanh(forget[i]) * x[i] + Math.Tanh(gain[i]) * m[i];
        return y;
    }

    /// <summary>
    /// The causal short convolution with its residual (llama.cpp's convolve): ggml_ssm_conv over [history; x], tap k of
    /// channel c at w[c * taps + k] multiplying x[t - (taps - 1 - k)] (tap taps - 1 takes the current token); with
    /// dynamic taps (global K and V), plus dyn[t][h * taps + L] * x[t - L] per head h; then (x + conv) /
    /// sqrt(taps * (dyn ? 2 : 1) + 1). <paramref name="history"/> becomes the last taps - 1 inputs.
    /// </summary>
    private double[][] Convolve(double[][] x, ref double[][] history, float[] w, double[][] dyn, int headDim)
    {
        int taps = Cfg.Taps, past = taps - 1, T = x.Length, C = x[0].Length;
        var full = new double[past + T][];
        for (int i = 0; i < past; i++) full[i] = history[i];
        for (int t = 0; t < T; t++) full[past + t] = x[t];
        double norm = 1.0 / Math.Sqrt((dyn != null ? 2 : 1) * taps + 1);
        var y = new double[T][];
        for (int t = 0; t < T; t++)
        {
            var row = new double[C];
            for (int ch = 0; ch < C; ch++)
            {
                double s = 0;
                for (int kk = 0; kk < taps; kk++) s += w[ch * taps + kk] * full[t + kk][ch];
                if (dyn != null)
                {
                    int h = ch / headDim;
                    for (int lag = 0; lag < taps; lag++) s += dyn[t][h * taps + lag] * full[t + past - lag][ch];
                }
                row[ch] = (x[t][ch] + s) * norm;
            }
            y[t] = row;
        }
        var next = new double[past][];
        for (int i = 0; i < past; i++) next[i] = (double[])full[T + i].Clone();
        history = next;
        return y;
    }

    /// <summary>clip(x, scale, c) on one row: x *= scale (if given), then x *= min(1, c / (|x|_2 + 1e-6)). True when the
    /// row was scaled down.</summary>
    private static bool Clip(double[] x, int offset, int length, float[] scale, double threshold)
    {
        double ss = 0;
        for (int i = 0; i < length; i++)
        {
            if (scale != null) x[offset + i] *= scale[offset + i];
            ss += x[offset + i] * x[offset + i];
        }
        double factor = Math.Clamp(threshold / (Math.Sqrt(ss) + 1e-6), 0.0, 1.0);
        for (int i = 0; i < length; i++) x[offset + i] *= factor;
        return factor < 1;
    }

    private static void HeadRmsNorm(double[] x, int offset, int length, double eps)
    {
        double ss = 0;
        for (int i = 0; i < length; i++) ss += x[offset + i] * x[offset + i];
        double inv = 1.0 / Math.Sqrt(ss / length + eps);
        for (int i = 0; i < length; i++) x[offset + i] *= inv;
    }

    private static double[] RmsNorm(double[] x, float[] weight, double eps)
    {
        double ss = 0;
        foreach (double v in x) ss += v * v;
        double inv = 1.0 / Math.Sqrt(ss / x.Length + eps);
        var y = new double[x.Length];
        for (int i = 0; i < x.Length; i++) y[i] = x[i] * inv * (weight?[i] ?? 1.0);
        return y;
    }

    private static IEnumerable<double[]> BlocksAt(List<double[][]> blocks, int t) => blocks.Select(b => b[t]);

    private static double[] Add(double[] a, double[] b)
    {
        var y = new double[a.Length];
        for (int i = 0; i < a.Length; i++) y[i] = a[i] + b[i];
        return y;
    }

    private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

    private static double Dot(float[] w, int offset, double[] x)
    {
        double s = 0;
        for (int i = 0; i < x.Length; i++) s += w[offset + i] * x[i];
        return s;
    }

    /// <summary>W . x for a GGUF matrix of ne [x.Length, rows] starting at <paramref name="offset"/> (row r at offset + r *
    /// x.Length): a 3-D expert tensor's expert e starts at e * rows * x.Length.</summary>
    private static double[] MatVec(float[] w, int offset, double[] x, int rows)
    {
        var y = new double[rows];
        for (int r = 0; r < rows; r++) y[r] = Dot(w, offset + r * x.Length, x);
        return y;
    }

    private static double[][] Zeros(int rows, int columns)
    {
        var z = new double[rows][];
        for (int i = 0; i < rows; i++) z[i] = new double[columns];
        return z;
    }
}
