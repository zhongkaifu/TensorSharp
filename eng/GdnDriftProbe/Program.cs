// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// GdnDriftProbe: how far a gated delta-net layer's F32 state drifts from an F64 sequential recurrence over a long run of real
// activations (issue #391: the recurrence's row sums regrouped from 32 lanes a row to 8).
//
//   GdnDriftProbe MODEL TEXT OUT.tsv [--tokens N] [--layers 0,44] [--checkpoint 1024]
//
// The model (direct CUDA, a Qwen 3.5 family target) takes the text's first 8 tokens as its prompt, then the next N tokens as
// verify windows of 8 rows, every row kept (teacher-forced: the rows are the text's, whatever the model would draw). Each
// window's scan inputs (q, k, v, g, beta) of the named layers are read back after the verify, and their states after the commits
// at every checkpoint: the device's own recurrence (the verify windows' commits re-step each window's rows on the live state with
// the decode rows' kernel). From the state before the first window and the captured inputs the probe steps:
//   f64     the recurrence in F64 (the reference);
//   lanes32 the F32 recurrence with the 32-lane row sums (ts_graph_gdn_rows4's grouping: a lane's sum over keys lane + 32 j by
//           fused multiply-adds, then a five-step xor butterfly), MathF.Exp for the decay;
//   lanes8  the same with the 8-lane row sums (ts_graph_gdn8's: keys lane + 8 j, a three-step butterfly).
// Every state element takes the kernels' operations in their order (the decay, the delta from the decayed state, the update), so
// the two F32 steppers differ only by the sums' grouping and see the same inputs. Each checkpoint's row: the relative Frobenius
// distance of the device's, lanes32's and lanes8's states from the F64 one, per layer. MAX_CONTEXT must cover the run.
using System.Globalization;
using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Speculative;

if (args.Length < 3)
{
    Console.Error.WriteLine("Usage: GdnDriftProbe MODEL TEXT OUT.tsv [--tokens N] [--layers 0,44] [--checkpoint 1024]");
    return 2;
}
string modelPath = args[0], textPath = args[1], outPath = args[2];
int tokensWanted = 16384, checkpoint = 1024;
int[] layers = { 0, 44 };
for (int i = 3; i + 1 < args.Length; i += 2)
{
    switch (args[i])
    {
        case "--tokens": tokensWanted = int.Parse(args[i + 1], CultureInfo.InvariantCulture); break;
        case "--layers": layers = args[i + 1].Split(',').Select(s => int.Parse(s, CultureInfo.InvariantCulture)).ToArray(); break;
        case "--checkpoint": checkpoint = int.Parse(args[i + 1], CultureInfo.InvariantCulture); break;
        default: Console.Error.WriteLine($"unknown option {args[i]}"); return 2;
    }
}
const int Window = 8, Prompt = 8;
if (tokensWanted % Window != 0 || checkpoint % Window != 0)
{
    Console.Error.WriteLine($"--tokens and --checkpoint must be multiples of {Window}");
    return 2;
}

using var model = (StandardDecoderModel)ModelBase.Create(modelPath, BackendType.Cuda);
var unified = (IUnifiedBatchModel)model;
var batched = (IBatchedSpeculativeTarget)model;
List<int> encoded = model.Tokenizer.Encode(File.ReadAllText(textPath), addSpecial: false);
if (encoded.Count < Prompt + tokensWanted)
{
    Console.Error.WriteLine($"the text has {encoded.Count} tokens; the run needs {Prompt + tokensWanted}");
    return 2;
}
int[] text = encoded.Take(Prompt + tokensWanted).ToArray();
const string id = "drift";
unified.EnsureSequenceSlot(id, Prompt + tokensWanted + 64);
var lastLogits = new float[1][];
if (!unified.TryForwardUnified(new[] { id }, new[] { text[..Prompt] }, new[] { 0 }, lastLogits))
    throw new InvalidOperationException("the prompt step did not run");

var probes = new List<LayerProbe>();
foreach (int layer in layers)
{
    if (model.RecurrentStateForProbe(id, layer) is not Qwen35RuntimeModel.DeviceGdnState state)
    {
        Console.Error.WriteLine($"layer {layer} has no gated delta-net state on this model; skipped");
        continue;
    }
    probes.Add(new LayerProbe(layer, state, tokensWanted));
}
if (probes.Count == 0) return 2;
foreach (LayerProbe p in probes) p.State0 = LayerProbe.Read(p.State.Ssm);

var deviceStates = new Dictionary<(int Layer, int Tokens), float[]>();
var started = DateTime.UtcNow;
for (int done = 0; done < tokensWanted; done += Window)
{
    int position = Prompt + done;
    int[] window = text[position..(position + Window)];
    var rows = new float[(long)Window * model.Config.VocabSize];
    if (!batched.TryBatchedSpecVerifyLogits(new[] { id }, new[] { position }, Window, window, rows, null))
        throw new InvalidOperationException($"the verify window at {position} did not run");
    foreach (LayerProbe p in probes) p.Capture(done);
    batched.CommitBatchedSpecVerify(new[] { id }, new[] { position }, new[] { Window });
    int steps = done + Window;
    if (steps % checkpoint == 0)
    {
        foreach (LayerProbe p in probes) deviceStates[(p.Layer, steps)] = LayerProbe.Read(p.State.Ssm);
        Console.Error.WriteLine($"{steps} tokens ({(DateTime.UtcNow - started).TotalSeconds:F0} s)");
    }
}
model.OnSequenceReleased(id);

using var output = new StreamWriter(outPath);
output.WriteLine("layer\ttokens\tdevice\tlanes32\tlanes8\tref_norm");
foreach (LayerProbe p in probes)
{
    var reference = new Stepper(p, Stepper.Kind.F64);
    var lanes32 = new Stepper(p, Stepper.Kind.Lanes32);
    var lanes8 = new Stepper(p, Stepper.Kind.Lanes8);
    for (int steps = checkpoint; steps <= tokensWanted; steps += checkpoint)
    {
        Parallel.Invoke(() => reference.StepTo(steps), () => lanes32.StepTo(steps), () => lanes8.StepTo(steps));
        double norm = reference.Norm();
        string line = string.Join('\t', p.Layer.ToString(CultureInfo.InvariantCulture), steps.ToString(CultureInfo.InvariantCulture),
            (reference.Distance(deviceStates[(p.Layer, steps)]) / norm).ToString("E4", CultureInfo.InvariantCulture),
            (reference.Distance(lanes32.F32) / norm).ToString("E4", CultureInfo.InvariantCulture),
            (reference.Distance(lanes8.F32) / norm).ToString("E4", CultureInfo.InvariantCulture),
            norm.ToString("E4", CultureInfo.InvariantCulture));
        output.WriteLine(line);
        output.Flush();
        Console.WriteLine(line);
    }
}
return 0;

/// <summary>A layer's state and its captured scan inputs, token-major: q, k [tokens][hk][S], v [tokens][h][S], g, beta [tokens][h].</summary>
internal sealed class LayerProbe
{
    public readonly int Layer, Heads, KeyHeads, S, Tokens;
    public readonly Qwen35RuntimeModel.DeviceGdnState State;
    public readonly float[] Q, K, V, G, Beta;
    public float[] State0;

    public LayerProbe(int layer, Qwen35RuntimeModel.DeviceGdnState state, int tokens)
    {
        Layer = layer;
        State = state;
        Tokens = tokens;
        Heads = state.VHeads;
        S = (int)state.Ssm.Sizes[^1];
        KeyHeads = (state.Channels - Heads * S) / (2 * S);
        Q = new float[(long)tokens * KeyHeads * S];
        K = new float[Q.LongLength];
        V = new float[(long)tokens * Heads * S];
        G = new float[(long)tokens * Heads];
        Beta = new float[G.LongLength];
    }

    /// <summary>A device tensor's values (its device copy is the current one: a recorded step wrote it).</summary>
    public static float[] Read(Tensor t)
    {
        CudaFusedOps.TryMarkDeviceModified(t);
        return t.GetElementsAsFloat((int)t.ElementCount());
    }

    /// <summary>The last verify window's saved scan inputs (planar regions of the state's backing rows), its rows at
    /// <paramref name="at"/>.</summary>
    public void Capture(int at)
    {
        Tensor saved = State.SavedInputs ?? throw new InvalidOperationException($"layer {Layer}: the verify window saved no inputs");
        float[] all = Read(saved);
        int rows = State.BackingRows, window = 8;
        long qk = (long)rows * KeyHeads * S, vs = (long)rows * Heads * S, gs = (long)rows * Heads;
        long qkRow = (long)KeyHeads * S, vRow = (long)Heads * S;
        Array.Copy(all, 0, Q, at * qkRow, window * qkRow);
        Array.Copy(all, qk, K, at * qkRow, window * qkRow);
        Array.Copy(all, 2 * qk, V, at * vRow, window * vRow);
        Array.Copy(all, 2 * qk + vs, G, (long)at * Heads, window * Heads);
        Array.Copy(all, 2 * qk + vs + gs, Beta, (long)at * Heads, window * Heads);
    }
}

/// <summary>One recurrence over a layer's captured inputs from its state before the first window: state [h][S rows][S keys]
/// (row r of a head's state holds the keys' values of value row r, as the kernels keep it).</summary>
internal sealed class Stepper
{
    public enum Kind { F64, Lanes32, Lanes8 }

    private readonly LayerProbe _p;
    private readonly Kind _kind;
    private readonly double[] _f64;
    public readonly float[] F32;
    private int _at;

    public Stepper(LayerProbe p, Kind kind)
    {
        _p = p;
        _kind = kind;
        if (kind == Kind.F64) _f64 = p.State0.Select(v => (double)v).ToArray();
        else F32 = (float[])p.State0.Clone();
    }

    public void StepTo(int tokens)
    {
        int S = _p.S, h = _p.Heads, hk = _p.KeyHeads;
        for (; _at < tokens; _at++)
        {
            int t = _at;
            Parallel.For(0, h, head =>
            {
                int kh = head % hk;
                long qk = ((long)t * hk + kh) * S, vo = ((long)t * h + head) * S, gb = (long)t * h + head;
                if (_kind == Kind.F64) StepF64(head, qk, vo, gb);
                else StepF32(head, qk, vo, gb, _kind == Kind.Lanes8 ? 8 : 32);
            });
        }
    }

    private void StepF64(int head, long qk, long vo, long gb)
    {
        int S = _p.S;
        double e = Math.Exp(_p.G[gb]), b = _p.Beta[gb];
        for (int r = 0; r < S; r++)
        {
            long row = ((long)head * S + r) * S;
            double kv = 0;
            for (int c = 0; c < S; c++)
            {
                _f64[row + c] *= e;
                kv += _f64[row + c] * _p.K[qk + c];
            }
            double delta = (_p.V[vo + r] - kv) * b;
            for (int c = 0; c < S; c++) _f64[row + c] += _p.K[qk + c] * delta;
        }
    }

    /// <summary>The kernels' F32 step of one head: per value row, each lane's fused multiply-add sum over its keys (lane + lanes * j),
    /// the lanes' xor butterfly, the delta, the update by fused multiply-adds.</summary>
    private void StepF32(int head, long qk, long vo, long gb, int lanes)
    {
        int S = _p.S, per = S / lanes;
        float e = MathF.Exp(_p.G[gb]), b = _p.Beta[gb];
        Span<float> partial = stackalloc float[32], swap = stackalloc float[32];
        for (int r = 0; r < S; r++)
        {
            long row = ((long)head * S + r) * S;
            for (int lane = 0; lane < lanes; lane++)
            {
                float acc = 0f;
                for (int j = 0; j < per; j++)
                {
                    long c = row + lane + lanes * j;
                    F32[c] *= e;
                    acc = MathF.FusedMultiplyAdd(F32[c], _p.K[qk + lane + lanes * j], acc);
                }
                partial[lane] = acc;
            }
            for (int off = lanes / 2; off > 0; off >>= 1)
            {
                for (int lane = 0; lane < lanes; lane++) swap[lane] = partial[lane] + partial[lane ^ off];
                swap[..lanes].CopyTo(partial);
            }
            float kv = partial[0];
            float delta = (_p.V[vo + r] - kv) * b;
            for (int c = 0; c < S; c++)
                F32[row + c] = MathF.FusedMultiplyAdd(_p.K[qk + c], delta, F32[row + c]);
        }
    }

    public double Norm()
    {
        double s = 0;
        foreach (double v in _f64) s += v * v;
        return Math.Sqrt(s);
    }

    public double Distance(float[] state)
    {
        double s = 0;
        for (long i = 0; i < _f64.LongLength; i++)
        {
            double d = state[i] - _f64[i];
            s += d * d;
        }
        return Math.Sqrt(s);
    }
}
