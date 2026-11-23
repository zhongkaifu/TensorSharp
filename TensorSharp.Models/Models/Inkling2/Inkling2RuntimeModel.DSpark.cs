// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;

using TensorSharp.Graph;
using TensorSharp.Runtime;

namespace TensorSharp.Models
{
    /// <summary>
    /// Inkling 2's DSpark drafter (the checkpoint's mtp.safetensors as an inkling2_dspark GGUF, eng/inkling2/dspark_to_gguf.py):
    /// three dense draft layers over [anchor, mask x 7] and a recurrent Markov head that picks the seven drafts left to right.
    /// <list type="bullet">
    /// <item>Features: the stream's current slot entering visits 5, 23 and 41 and the head's input (Config.NumLayers), so a
    /// row's features are 4 x hidden (the trunk captures them, EmitCaptureRow).</item>
    /// <item>Context: each draft layer mixes a committed row's four features with its own weights (softmax(fuse_weight)), RMS-norms
    /// the mix (dflare_norm), projects it to K (per-head norm, NeoX rotary at theta 1e6) and V, and writes them at the row's
    /// position: a ring for the local layers (window 1024), a linear cache for the global one.</item>
    /// <item>Block: the anchor's normed embedding and the mask embedding through pre-norm layers whose attention sees the
    /// context before the block (the local layers its last 1024 positions) and the whole block (non-causal); the head is the
    /// target's LM head over the mask rows plus the Markov bias: a gated recurrence over [state, w1[previous]] and the row's
    /// [hidden, f_rnn] context, folded through the LM head ([vocab, rank]); argmax picks each draft and feeds the next step.</item>
    /// </list>
    /// </summary>
    public partial class Inkling2RuntimeModel
    {
        private sealed class DSpark : FamilyBlockDrafter
        {
            private const string Prefix = "dspark.";
            private readonly Inkling2RuntimeModel _m;
            private readonly int _layers, _hidden, _heads, _kvHeads, _headDim, _ffn, _window, _block, _rank, _ringRows, _capacity;
            private readonly float _eps;
            private readonly bool[] _local;
            private readonly int[] _targets;
            private readonly float[,] _alpha;   // [row, tap]: rows 0 .. layers - 1 the draft layers', the last the Markov head's
            private readonly RotarySpec _rotary;
            private readonly Dictionary<string, Context> _contexts = new(StringComparer.Ordinal);
            private Context _active;
            private Tensor _tokenIds;

            /// <summary>One request's context: K and V a draft layer (F16 [kvHeads, rows, headDim]); the global layers hold
            /// Capacity rows (grown on demand), the local ones the ring.</summary>
            private sealed class Context
            {
                public Tensor[] K, V;
                public int Capacity;
                // Committed rows not in the caches yet (a short catch-up waits for the next draft): their first position and
                // features, contiguous.
                public int PendingFirst = -1, PendingRows;
                public float[] Pending;
            }

            /// <summary>TS_DSPARK_LONE_MERGE=0: a lone draft injects the waiting rows in a program of their own first.</summary>
            private static readonly bool LoneMergeDisabled = Environment.GetEnvironmentVariable("TS_DSPARK_LONE_MERGE") == "0";

            /// <summary>TS_DSPARK_DEFER=0: every catch-up runs its own inject program.</summary>
            private static readonly bool DeferDisabled = Environment.GetEnvironmentVariable("TS_DSPARK_DEFER") == "0";

            /// <summary>Rows a context's global layers start with, and released contexts kept for the next requests.</summary>
            private const int PooledContexts = 2;
            /// <summary>The global layers' first rows; grown by doubling (EnsureCapacity).</summary>
            private const int InitialCapacity = 8192;
            private readonly Stack<Context> _pool = new();

            public DSpark(Inkling2RuntimeModel m, string path)
            {
                _m = m;
                using var gguf = new GgufFile(path);
                string a = gguf.GetString("general.architecture");
                if (a != "inkling2_dspark") throw new InvalidOperationException($"{path} is a '{a}' GGUF, not an Inkling 2 DSpark drafter.");
                _layers = (int)gguf.GetUint32($"{a}.block_count");
                _hidden = (int)gguf.GetUint32($"{a}.embedding_length");
                _heads = (int)gguf.GetUint32($"{a}.attention.head_count");
                _kvHeads = (int)gguf.GetUint32($"{a}.attention.head_count_kv");
                _headDim = (int)gguf.GetUint32($"{a}.attention.key_length");
                _ffn = (int)gguf.GetUint32($"{a}.feed_forward_length");
                _eps = gguf.GetFloat32($"{a}.attention.layer_norm_rms_epsilon");
                _window = (int)gguf.GetUint32($"{a}.attention.sliding_window");
                _block = (int)gguf.GetUint32($"{a}.block_size");
                _rank = (int)gguf.GetUint32($"{a}.markov_rank");
                _local = gguf.GetBoolArray($"{a}.attention.sliding_window_pattern");
                _targets = gguf.GetInt32Array($"{a}.target_layers");
                _rotary = new RotarySpec(_headDim, neoX: true, gguf.GetFloat32($"{a}.rope.freq_base"));
                if (_hidden != m.Config.HiddenSize)
                    throw new InvalidOperationException($"DSpark hidden size {_hidden} != the target's {m.Config.HiddenSize}.");
                if (_targets == null || _targets.Length == 0 || _local == null || _local.Length != _layers)
                    throw new InvalidOperationException("DSpark drafter GGUF lacks its target layers or its sliding window pattern.");
                foreach (int t in _targets)
                    if (t < 0 || t > m.Config.NumLayers)
                        throw new InvalidOperationException($"DSpark target layer {t} is outside the target's {m.Config.NumLayers} runs.");
                m.LoadDSparkTensors(gguf);
                float[] alpha = m._weights[Prefix + "fuse_alpha"].GetElementsAsFloat(_targets.Length * (_layers + 1));
                _alpha = new float[_layers + 1, _targets.Length];
                for (int r = 0; r <= _layers; r++)
                    for (int t = 0; t < _targets.Length; t++)
                        _alpha[r, t] = alpha[r * _targets.Length + t];
                // A ring keeps the window, a block and a verify's rows ahead of it.
                _ringRows = (_window + 2 * _block + 64 + 255) / 256 * 256;
                int context = m.MaxContextLength > 0 ? m.MaxContextLength : 32768;
                _capacity = (context + _block + 255) / 256 * 256;
                Console.WriteLine($"  DSpark drafter: {_layers} layers (local {string.Join(",", Array.ConvertAll(_local, l => l ? "1" : "0"))}), " +
                    $"block {_block}, Markov rank {_rank}, targets [{string.Join(",", _targets)}], ring {_ringRows}, context {_capacity}");
            }

            private static readonly bool Debug = Environment.GetEnvironmentVariable("TS_DSPARK_DEBUG") == "1";
            private static string _dump = Environment.GetEnvironmentVariable("TS_DSPARK_DUMP");

            private void DumpRows(float[] hRows, int rowOffset, int n, int first)
            {
                if (_dump == null) return;
                int feat = _targets.Length * _hidden;
                System.IO.Directory.CreateDirectory(_dump);
                using var fs = new System.IO.FileStream(System.IO.Path.Combine(_dump, "ctx.bin"), System.IO.FileMode.Append);
                using var w = new System.IO.BinaryWriter(fs);
                for (int r = 0; r < n; r++)
                {
                    w.Write(first + r);
                    for (int i = 0; i < feat; i++) w.Write(hRows[(long)(rowOffset + r) * feat + i]);
                }
            }

            public override int DraftTokens => _block - 1;

            public override int[] CaptureLayers => _targets;

            /// <summary>The sliding rings take a prompt in steps of this many rows, and the trunk returns features of a step
            /// at most that long.</summary>
            public override int PrefillChunkCap => _m.RingStepRowsInUse;

            private int Rows(Context c, int layer) => _local[layer] ? _ringRows : c.Capacity;

            private int Cell(int layer, int position) => _local[layer] ? position % _ringRows : position;

            private Context Active
            {
                get
                {
                    if (_active != null) return _active;
                    Bind(null);
                    return _active;
                }
            }

            public override void Bind(string requestId)
            {
                string key = requestId ?? "";
                if (!_contexts.TryGetValue(key, out Context c))
                {
                    // A pooled context's rows are an earlier request's (finite, and masked until this one writes them).
                    if (!_pool.TryPop(out c))
                    {
                        c = new Context { K = new Tensor[_layers], V = new Tensor[_layers], Capacity = Math.Min(_capacity, InitialCapacity) };
                        for (int l = 0; l < _layers; l++)
                        {
                            c.K[l] = new Tensor(_m._allocator, DType.Float16, _kvHeads, Rows(c, l), _headDim);
                            c.V[l] = new Tensor(_m._allocator, DType.Float16, _kvHeads, Rows(c, l), _headDim);
                            _m.InitializeCacheTensor(c.K[l]);
                            _m.InitializeCacheTensor(c.V[l]);
                        }
                    }
                    _contexts[key] = c;
                }
                _active = c;
            }

            /// <summary>The global layers' rows grown to hold position <paramref name="needed"/> - 1: doubled (at least), the
            /// rows held copied, the graphs over the old caches dropped.</summary>
            private void EnsureCapacity(Context c, int needed)
            {
                if (needed <= c.Capacity || c.Capacity >= _capacity) return;
                int grown = Math.Min(_capacity, (Math.Max(needed, 2 * c.Capacity) + 1023) / 1024 * 1024);
                _m.ForgetFamilyDraftCaches(c.K);
                for (int l = 0; l < _layers; l++)
                {
                    if (_local[l]) continue;
                    c.K[l] = Grow(c.K[l], c.Capacity, grown);
                    c.V[l] = Grow(c.V[l], c.Capacity, grown);
                }
                if (Debug) Console.Error.WriteLine($"[dspark] context grown {c.Capacity} -> {grown} rows");
                c.Capacity = grown;
            }

            private Tensor Grow(Tensor old, int rows, int grown)
            {
                var t = new Tensor(_m._allocator, DType.Float16, _kvHeads, grown, _headDim);
                _m.InitializeCacheTensor(t);
                // The graphs wrote the rows in device memory: bring them to the host tensor before copying it.
                _m.SyncTensorHostCache(old);
                using (Tensor src = old.Narrow(1, 0, rows))
                using (Tensor dst = t.Narrow(1, 0, rows))
                    Ops.Copy(dst, src);
                _m.HostWritten(t);
                _m.InvalidateTensorDeviceCache(old);
                old.Dispose();
                return t;
            }

            public override void Release(string requestId)
            {
                string key = requestId ?? "";
                if (!_contexts.Remove(key, out Context c)) return;
                if (ReferenceEquals(_active, c)) _active = null;
                c.PendingRows = 0;
                c.PendingFirst = -1;
                if (_pool.Count < PooledContexts) _pool.Push(c);
                else Free(c);
            }

            private void Free(Context c)
            {
                _m.ForgetFamilyDraftCaches(c.K);
                foreach (Tensor t in c.K) t.Dispose();
                foreach (Tensor t in c.V) t.Dispose();
            }

            public override void Dispose()
            {
                foreach (Context c in _contexts.Values) Free(c);
                _contexts.Clear();
                while (_pool.TryPop(out Context pooled)) Free(pooled);
                _active = null;
                _tokenIds?.Dispose();
                _tokenIds = null;
            }

            // ------------------------------------------------------------ the context

            public override unsafe void CatchUp(int[] tokens, float[] hRows, int startPos)
            {
                if (hRows == null || tokens == null || tokens.Length == 0) return;
                int feat = _targets.Length * _hidden;
                int n = (int)Math.Min(tokens.Length, hRows.LongLength / feat);
                // Row k holds the features of position startPos - 1 + k; position -1 (the first prompt row's carry) has none.
                int skip = startPos == 0 ? 1 : 0;
                // A ring layer keeps its last ring rows: the global layer takes every row, so all rows go in.
                int rows = n - skip;
                if (Debug)
                {
                    double norm = 0;
                    for (int i = 0; i < feat && skip < n; i++) norm += (double)hRows[(long)skip * feat + i] * hRows[(long)skip * feat + i];
                    Console.Error.WriteLine($"[dspark] catch-up tokens={tokens.Length} rows={rows} first={startPos - 1 + skip} feat0 |x|={Math.Sqrt(norm):F2}");
                }
                if (rows <= 0) return;
                DumpRows(hRows, skip, rows, startPos - 1 + skip);
                Context c = Active;
                int first = startPos - 1 + skip;
                // A short catch-up waits for the next draft when it continues the waiting rows (at most a block of them).
                if (!DeferDisabled && rows <= _block && (c.PendingRows == 0 || c.PendingFirst + c.PendingRows == first)
                    && c.PendingRows + rows <= _block)
                {
                    c.Pending ??= new float[(long)_block * feat];
                    if (c.PendingRows == 0) c.PendingFirst = first;
                    Array.Copy(hRows, (long)skip * feat, c.Pending, (long)c.PendingRows * feat, (long)rows * feat);
                    c.PendingRows += rows;
                    return;
                }
                Flush(c);
                Inject(hRows, skip, rows, first);
            }

            /// <summary>The context's waiting rows into its caches (an inject program of their own).</summary>
            private void Flush(Context c)
            {
                if (c.PendingRows == 0) return;
                int n = c.PendingRows, first = c.PendingFirst;
                c.PendingRows = 0;
                c.PendingFirst = -1;
                Inject(c.Pending, 0, n, first);
            }

            private unsafe void Inject(float[] hRows, int rowOffset, int n, int first)
            {
                int feat = _targets.Length * _hidden;
                var features = new Tensor(_m._allocator, DType.Float32, n, feat);
                long bytes = (long)n * feat * sizeof(float);
                fixed (float* src = &hRows[(long)rowOffset * feat])
                    Buffer.MemoryCopy(src, GetFloatPtrForOverwrite(features), bytes, bytes);
                _m.InvalidateTensorDeviceCache(features);
                Context c = Active;
                EnsureCapacity(c, first + n);
                StandardDecoderModel.CachedGraph graph = null;
                bool owned = false;
                try
                {
                    graph = _m.RunFamilyDraftProgram(c.K, c.V, c.Capacity, n, first,
                        step => EmitInject(step, c, features, n), out _, out owned, "inkling2-dspark-inject");
                    if (graph == null) throw new InvalidOperationException("the DSpark inject program could not be traced");
                }
                finally
                {
                    if (owned) graph?.Dispose();
                    features.Dispose();
                }
            }

            /// <summary>Per draft layer, the context K and V of <paramref name="n"/> feature rows written at their positions.</summary>
            private StandardDecoderModel.StepOutputs EmitInject(GraphStep step, Context c, Tensor features, int n)
            {
                var positions = new int[n];
                for (int i = 0; i < n; i++) positions[i] = step.Starts[0] + i;
                step.SetPositions(positions);
                int x = step.HiddenInput(features, (long)_targets.Length * _hidden, n);
                for (int l = 0; l < _layers; l++)
                    EmitContextRows(step, c, l, x, n, positions);
                return new StandardDecoderModel.StepOutputs { Logits = -1, Final = -1, Hidden = -1 };
            }

            /// <summary>Layer <paramref name="l"/>'s mix of the feature rows <paramref name="x"/> [taps * hidden, n] (row-major).</summary>
            private int Mix(GraphStep step, int x, int n, int row, string norm)
            {
                const long f = sizeof(float);
                long width = (long)_targets.Length * _hidden;
                int mixed = -1;
                for (int t = 0; t < _targets.Length; t++)
                {
                    int tap = step.View2(x, _hidden, n, width * f, t * (long)_hidden * f);
                    int scaled = step.Op(GraphOp.Scale, stackalloc[] { tap }, default, stackalloc[] { _alpha[row, t] });
                    mixed = mixed < 0 ? scaled : step.Op(GraphOp.Add, mixed, scaled);
                }
                return step.RmsNorm(mixed, step.Weight(norm), _eps);
            }

            private void EmitContextRows(GraphStep step, Context c, int l, int x, int n, int[] positions)
            {
                string p = $"{Prefix}blk.{l}.";
                int g = Mix(step, x, n, l, p + "dflare_norm");
                int kv = step.MatMul(p + "attn_kv_ctx", g);                                                   // [2 kv, n]
                long kvw = (long)_kvHeads * _headDim;
                int k = step.Reshape(step.Cont(step.View2(kv, kvw, n, 2 * kvw * sizeof(float), 0)), _headDim, _kvHeads, n);
                int v = step.Reshape(step.Cont(step.View2(kv, kvw, n, 2 * kvw * sizeof(float), kvw * sizeof(float))), _headDim, _kvHeads, n);
                k = step.Rope(step.RmsNorm(k, step.Weight(p + "attn_k_ctx_norm"), _eps), _rotary);
                Write(step, c, l, k, v, n, positions);
            }

            /// <summary>K and V [head_dim, kv_heads, n] at <paramref name="positions"/>' cells of layer <paramref name="l"/>;
            /// returns the written K (reading it orders the read after the write) and V through <paramref name="vOut"/>.</summary>
            private int Write(GraphStep step, Context c, int l, int k, int v, int n, int[] positions, Action<int> vOut = null)
            {
                int rows = Rows(c, l);
                int ck = step.State(c.K[l], GraphFormat.F16, _headDim, rows, _kvHeads);
                int cv = step.State(c.V[l], GraphFormat.F16, _headDim, rows, _kvHeads);
                var cells = new int[n];
                for (int i = 0; i < n; i++) cells[i] = Cell(l, positions[i]);
                int at = step.InputRows(cells);
                int kRows = n == 1 ? step.Reshape(k, _headDim, 1, _kvHeads) : step.Cont(step.Permute(k, 0, 2, 1, 3));
                int vRows = n == 1 ? step.Reshape(v, _headDim, 1, _kvHeads) : step.Cont(step.Permute(v, 0, 2, 1, 3));
                int wk = step.Op(GraphOp.SetRows, ck, kRows, at);
                int wv = step.Op(GraphOp.SetRows, cv, vRows, at);
                step.Effect(wk);
                step.Effect(wv);
                vOut?.Invoke(wv);
                return wk;
            }

            // ------------------------------------------------------------ the block

            public override int DraftBlock(int lastToken, float[] hPrev, int position, int[] draftOut, float[] confOut)
            {
                int gamma = Math.Min(_block - 1, draftOut.Length);
                if (gamma < 1 || hPrev == null || position < 1) return 0;
                int feat = _targets.Length * _hidden;
                Context c = Active;
                // The waiting rows go in with the carry, in this program (its context rows: the waiting rows, padding, the carry),
                // when they end at the carry; others in a program of their own first.
                bool merge = !LoneMergeDisabled;
                if (!merge || (c.PendingRows > 0 && c.PendingFirst + c.PendingRows != position)) Flush(c);
                EnsureCapacity(c, position + _block + 1);
                Tensor embedded = _m.Embedding(new[] { lastToken });
                int[] ctxPos = null;
                Tensor carry;
                if (merge)
                {
                    int cr = _block + 1;
                    ctxPos = new int[cr];
                    var rowsHost = new float[(long)cr * feat];
                    int waiting = c.PendingRows;
                    if (waiting > 0) Array.Copy(c.Pending, 0, rowsHost, 0, (long)waiting * feat);
                    for (int i = 0; i < waiting; i++) ctxPos[i] = c.PendingFirst + i;
                    // The padding: zero rows at the cell of position p + block, which no draft reads before it is written.
                    for (int i = waiting; i < _block; i++) ctxPos[i] = position + _block;
                    Array.Copy(hPrev, 0, rowsHost, (long)_block * feat, feat);
                    ctxPos[_block] = position - 1;
                    carry = new Tensor(_m._allocator, DType.Float32, cr, feat);
                    carry.SetElementsAsFloat(rowsHost);
                }
                else
                {
                    carry = new Tensor(_m._allocator, DType.Float32, 1, feat);
                    carry.SetElementsAsFloat(hPrev.AsSpan(0, feat).ToArray());
                }
                _m.HostWritten(carry);
                StandardDecoderModel.CachedGraph graph = null;
                bool owned = false;
                try
                {
                    graph = _m.RunFamilyDraftProgram(c.K, c.V, c.Capacity, _block, position,
                        step => EmitBlock(step, c, embedded, carry, lastToken, position, ctxPos), out StandardDecoderModel.StepOutputs outputs, out owned,
                        "inkling2-dspark-block");
                    if (graph == null) return 0;
                    if (merge) { c.PendingRows = 0; c.PendingFirst = -1; }   // the waiting rows are in the caches now
                    for (int i = 0; i < gamma; i++)
                    {
                        draftOut[i] = (int)graph.Outputs[outputs.Draft[i]][0];
                        if (confOut != null && i < confOut.Length) confOut[i] = graph.Outputs[outputs.Draft[_block - 1 + i]][0];
                    }
                }
                finally
                {
                    if (owned) graph?.Dispose();
                    embedded.Dispose();
                    carry.Dispose();
                }
                if (_dump != null)
                {
                    DumpRows(hPrev, 0, 1, position - 1);
                    System.IO.File.WriteAllText(System.IO.Path.Combine(_dump, "block.txt"),
                        $"{position} {lastToken} {string.Join(",", draftOut[..gamma])} {string.Join(",", Array.ConvertAll(confOut[..gamma], v => v.ToString("R")))}");
                    _dump = null;
                }
                if (Debug)
                {
                    double norm = 0;
                    for (int i = 0; i < feat; i++) norm += (double)hPrev[i] * hPrev[i];
                    Console.Error.WriteLine($"[dspark] block p={position} anchor={lastToken} carry |x|={Math.Sqrt(norm):F2} drafts=[{string.Join(",", draftOut[..gamma])}] " +
                        $"conf=[{string.Join(",", Array.ConvertAll(confOut[..gamma], v => v.ToString("F2")))}]");
                }
                return gamma;
            }

            /// <summary>The carry's context rows at position - 1, the block's layers at positions p .. p + block - 1, the head and
            /// the Markov walk. Outputs (<see cref="StandardDecoderModel.StepOutputs.Draft"/>): the drafts as F32 ids, then their
            /// confidences.</summary>
            private StandardDecoderModel.StepOutputs EmitBlock(GraphStep step, Context c, Tensor embedded, Tensor carry, int anchor, int p,
                int[] ctxPos)
            {
                int b = _block, h = _hidden, vocab = _m.Config.VocabSize;
                const long f = sizeof(float);
                int x0;
                if (ctxPos == null)
                {
                    // The carry (features of position p - 1) goes into the context first.
                    var carryPos = new[] { p - 1 };
                    step.SetPositions(carryPos);
                    x0 = step.HiddenInput(carry, (long)_targets.Length * h, 1);
                    for (int l = 0; l < _layers; l++)
                        EmitContextRows(step, c, l, x0, 1, carryPos);
                }
                else
                {
                    // The context rows (the waiting rows, padding, the carry at p - 1) first; the carry (row b) for the head.
                    step.SetPositions(ctxPos);
                    int xc = step.HiddenInput(carry, (long)_targets.Length * h, ctxPos.Length);
                    for (int l = 0; l < _layers; l++)
                        EmitContextRows(step, c, l, xc, ctxPos.Length, ctxPos);
                    x0 = step.Op(GraphOp.GetRows, xc, step.InputI32(new[] { b }, 1));                            // [feat, 1]
                }
                var positions = new int[b];
                for (int i = 0; i < b; i++) positions[i] = p + i;
                step.SetPositions(positions);

                int emb = step.RmsNorm(step.HiddenInput(embedded, h, 1), step.Weight("token_embd_norm.weight"), _m.Config.Eps);
                int masks = step.Op(GraphOp.Repeat, stackalloc[] { step.Reshape(step.Weight(Prefix + "mask_embed"), h, 1),
                    step.ConstantInputF32(new float[(long)h * (b - 1)], h, b - 1) });
                int x = step.Op(GraphOp.Concat, stackalloc[] { emb, masks }, stackalloc long[] { 1 });        // [h, b]
                float scale = 1f / MathF.Sqrt(_headDim);
                long qw = (long)_heads * _headDim, kw = (long)_kvHeads * _headDim, qkvw = qw + 2 * kw;
                for (int l = 0; l < _layers; l++)
                {
                    string pre = $"{Prefix}blk.{l}.";
                    int hn = step.RmsNorm(x, step.Weight(pre + "attn_norm"), _eps);
                    int qkv = step.MatMul(pre + "attn_qkv", hn);                                             // [q + 2 kv, b]
                    int q = step.Reshape(step.Cont(step.View2(qkv, qw, b, qkvw * f, 0)), _headDim, _heads, b);
                    int k = step.Reshape(step.Cont(step.View2(qkv, kw, b, qkvw * f, qw * f)), _headDim, _kvHeads, b);
                    int v = step.Reshape(step.Cont(step.View2(qkv, kw, b, qkvw * f, (qw + kw) * f)), _headDim, _kvHeads, b);
                    q = step.Rope(step.RmsNorm(q, step.Weight(pre + "attn_q_norm"), _eps), _rotary);
                    k = step.Rope(step.RmsNorm(k, step.Weight(pre + "attn_k_norm"), _eps), _rotary);
                    int wv = -1;
                    int wk = Write(step, c, l, k, v, b, positions, node => wv = node);
                    int keys, mask;
                    if (_local[l])
                    {
                        keys = _ringRows;
                        mask = LocalMask(step, p);
                    }
                    else
                    {
                        // The global layer reads the cache's positions up to the block's end, in buckets of 1024 (the mask hides
                        // the rest), so the program keeps its shape for a while.
                        keys = Math.Min(c.Capacity, (p + b + 1023) / 1024 * 1024);
                        mask = GlobalMask(step, p, keys);
                    }
                    long rowBytes = (long)_headDim * 2;
                    int kv = step.View3(wk, _headDim, keys, _kvHeads, rowBytes, Rows(c, l) * rowBytes, 0);
                    int vv = step.View3(wv, _headDim, keys, _kvHeads, rowBytes, Rows(c, l) * rowBytes, 0);
                    int fa = step.FlashAttention(step.Permute(q, 0, 2, 1, 3), kv, vv, mask, scale);
                    int attn = step.MatMul(pre + "attn_output", step.Reshape(fa, qw, b));
                    x = step.Op(GraphOp.Add, attn, x);
                    int fn = step.RmsNorm(x, step.Weight(pre + "ffn_norm"), _eps);
                    int gu = step.MatMul(pre + "ffn_gate_up", fn);                                              // [2 ffn, b]
                    int act = step.Op(GraphOp.Swiglu, gu);
                    x = step.Op(GraphOp.Add, step.MatMul(pre + "ffn_down", act), x);
                }
                int final = step.RmsNorm(x, step.Weight(Prefix + "output_norm"), _eps);
                int gamma = b - 1;
                int rows = step.Cont(step.View2(final, h, gamma, h * f, h * f));                              // the mask rows
                int baseLogits = step.MatMul("output.weight", rows);                                            // [vocab padded, gamma]

                // The Markov head's per-row context: rnn_ctx [hidden_i; f_rnn] + bias.
                int frnn = Mix(step, x0, 1, _layers, Prefix + "markov.dflare_norm");
                int frnnRows = step.Op(GraphOp.Repeat, stackalloc[] { frnn, step.ConstantInputF32(new float[(long)h * gamma], h, gamma) });
                int ctxIn = step.Op(GraphOp.Concat, stackalloc[] { rows, frnnRows }, stackalloc long[] { 0 });     // [2 h, gamma]
                int ctx = step.Op(GraphOp.Add, step.MatMul(Prefix + "markov.rnn_ctx", ctxIn), step.Weight(Prefix + "markov.rnn_bias"));
                int r = _rank;
                int state = step.ConstantInputF32(new float[r], r, 1);
                int prev = step.InputI32(new[] { anchor }, 1);
                int ids = TokenIds(step);
                var idOut = new int[gamma];
                var confOut = new int[gamma];
                for (int i = 0; i < gamma; i++)
                {
                    int w1 = step.Op(GraphOp.GetRows, step.Weight(Prefix + "markov.w1"), prev);              // [r, 1]
                    int sin = step.Op(GraphOp.Concat, stackalloc[] { state, w1 }, stackalloc long[] { 0 });     // [2 r, 1]
                    int raw = step.Op(GraphOp.Add, step.MatMul(Prefix + "markov.rnn_state", sin),
                        step.View2(ctx, 3L * r, 1, 3L * r * f, i * 3L * r * f));                                // [3 r, 1]
                    int gate = step.Op(GraphOp.Sigmoid, step.Cont(step.View2(raw, r, 1, r * f, 0)));
                    int cand = step.Op(GraphOp.Tanh, step.Cont(step.View2(raw, r, 1, r * f, r * f)));
                    int latent = step.Op(GraphOp.Tanh, step.Cont(step.View2(raw, r, 1, r * f, 2L * r * f)));
                    int keep = step.Op(GraphOp.Mul, gate, state);
                    int fresh = step.Op(GraphOp.Mul, step.Op(GraphOp.ScaleBias, stackalloc[] { gate }, default, stackalloc[] { -1f, 1f }), cand);
                    state = step.Op(GraphOp.Add, keep, fresh);
                    long padded = _m._vocabPadded;
                    int foldY = step.MatMul(Prefix + "markov.fold", latent);                                     // [padded, 1]
                    int foldB = step.Weight(Prefix + "markov.fold_bias");
                    int baseV = step.View2(baseLogits, padded, 1, padded * f, i * padded * f);
                    step.BeginFused(GraphFusion.InklingDraftPick, stackalloc[] { baseV, foldY, foldB }, stackalloc long[] { vocab },
                        stackalloc[] { _m._softcap });
                    int bias = step.Op(GraphOp.Add, foldY, foldB);
                    int logits = step.Op(GraphOp.Add, baseV, bias);
                    int real = step.Cont(step.View2(logits, vocab, 1, vocab * f, 0));                          // [vocab, 1]
                    int best = step.Op(GraphOp.Argmax, real);                                                    // I32 [1]
                    // The confidence is the softcapped softmax at the pick (the checkpoint's head transform).
                    int capped = step.Op(GraphOp.Scale, stackalloc[] { step.Op(GraphOp.Tanh, step.Op(GraphOp.Scale, stackalloc[] { real }, default,
                        stackalloc[] { 1f / _m._softcap })) }, default, stackalloc[] { _m._softcap });
                    int probs = step.Op(GraphOp.SoftMaxExt, stackalloc[] { capped }, default, stackalloc[] { 1f, 0f });
                    int conf = step.Op(GraphOp.GetRows, step.Reshape(probs, 1, vocab), best);
                    int idf = step.Op(GraphOp.GetRows, ids, best);
                    step.EndFused(stackalloc[] { best, conf, idf });
                    confOut[i] = step.Output(conf, 1);
                    idOut[i] = step.Output(idf, 1);
                    prev = best;
                }
                var draft = new int[2 * gamma];
                idOut.CopyTo(draft, 0);
                confOut.CopyTo(draft, gamma);
                return new StandardDecoderModel.StepOutputs { Logits = -1, Final = -1, Hidden = -1, Draft = draft };
            }

            // ------------------------------------------------------------ several requests' blocks at once

            public override int BatchDraftWidth => BatchedLanesDisabled ? 0 : _block - 1;

            /// <summary>The lanes' head walks and logits only as far as the windows verify (TS_DSPARK_NARROW=0: the whole
            /// block): four requests' 4-row windows take 3 of the 7 drafts.</summary>
            public override bool DraftsNarrowBatches => !NarrowDisabled;

            private static readonly bool NarrowDisabled = Environment.GetEnvironmentVariable("TS_DSPARK_NARROW") == "0";

            /// <summary>TS_DSPARK_LANES=0: every request drafts on its own.</summary>
            private static readonly bool BatchedLanesDisabled = Environment.GetEnvironmentVariable("TS_DSPARK_LANES") == "0";

            /// <summary>Every lane's block in one program: the draft layers' products, the head and the Markov walk run once
            /// over all lanes' rows (their weights read once), each lane's rows written into and attending its own context.</summary>
            public override bool DraftBatch(IReadOnlyList<string> requestIds, int[] anchors, float[][] carries, int[] positions,
                int[][] tokensOut, float[][] confOut)
            {
                int m = requestIds.Count, feat = _targets.Length * _hidden;
                if (m < 2) return false;
                // The drafts the blocks hold (the windows' drafts, at most the block's).
                int gamma = Math.Min(_block - 1, tokensOut[0].Length);
                if (gamma < 1) return false;
                for (int j = 0; j < m; j++)
                    if (carries[j] == null || carries[j].Length < feat || positions[j] < 1 || tokensOut[j].Length < gamma || confOut[j].Length < gamma)
                        return false;
                var cs = new Context[m];
                var ks = new Tensor[m][];
                var vs = new Tensor[m][];
                var caps = new int[m];
                int ctxRows = _block + 1;   // a lane's context rows: its waiting rows, padding, its carry
                var ctxPos = new int[m * ctxRows];
                for (int j = 0; j < m; j++)
                {
                    Bind(requestIds[j]);
                    cs[j] = _active;
                    // Rows that do not end at the carry (not this draft's) go in on their own first.
                    if (cs[j].PendingRows > 0 && cs[j].PendingFirst + cs[j].PendingRows != positions[j]) Flush(cs[j]);
                    EnsureCapacity(cs[j], positions[j] + _block + 1);
                    ks[j] = cs[j].K;
                    vs[j] = cs[j].V;
                    caps[j] = cs[j].Capacity;
                }
                Tensor embedded = _m.Embedding(anchors);
                var carry = new Tensor(_m._allocator, DType.Float32, m * ctxRows, feat);
                var rowsHost = new float[(long)m * ctxRows * feat];
                for (int j = 0; j < m; j++)
                {
                    Context c = cs[j];
                    int at = j * ctxRows, waiting = c.PendingRows;
                    if (waiting > 0) Array.Copy(c.Pending, 0, rowsHost, (long)at * feat, (long)waiting * feat);
                    for (int i = 0; i < waiting; i++) ctxPos[at + i] = c.PendingFirst + i;
                    // The padding: zero rows at the cell of position p + block, which no draft reads before it is written.
                    for (int i = waiting; i < _block; i++) ctxPos[at + i] = positions[j] + _block;
                    Array.Copy(carries[j], 0, rowsHost, (long)(at + _block) * feat, feat);
                    ctxPos[at + _block] = positions[j] - 1;
                }
                carry.SetElementsAsFloat(rowsHost);
                _m.HostWritten(carry);
                StandardDecoderModel.CachedGraph graph = null;
                bool owned = false;
                try
                {
                    graph = _m.RunFamilyDraftLanes(ks, vs, caps, _block, positions,
                        step => EmitBlocks(step, cs, embedded, carry, anchors, positions, ctxPos, gamma), out StandardDecoderModel.StepOutputs outputs, out owned,
                        "inkling2-dspark-lanes");
                    if (graph == null) return false;
                    // The lanes' waiting rows are in their caches now.
                    foreach (Context c in cs) { c.PendingRows = 0; c.PendingFirst = -1; }
                    for (int i = 0; i < gamma; i++)
                    {
                        float[] idsOut = graph.Outputs[outputs.Draft[i]], conf = graph.Outputs[outputs.Draft[gamma + i]];
                        for (int j = 0; j < m; j++)
                        {
                            tokensOut[j][i] = (int)idsOut[j];
                            confOut[j][i] = conf[j];
                        }
                    }
                    return true;
                }
                finally
                {
                    if (owned) graph?.Dispose();
                    embedded.Dispose();
                    carry.Dispose();
                }
            }

            private StandardDecoderModel.StepOutputs EmitBlocks(GraphStep step, Context[] cs, Tensor embedded, Tensor carry, int[] anchors, int[] ps,
                int[] ctxPos, int gamma)
            {
                int m = cs.Length, b = _block, h = _hidden, vocab = _m.Config.VocabSize;
                const long f = sizeof(float);
                long qw = (long)_heads * _headDim, kw = (long)_kvHeads * _headDim, qkvw = qw + 2 * kw;

                // Each lane's context rows (its waiting rows, padding, its carry at p - 1) go into its context first.
                int cr = b + 1, cn = m * cr;
                step.SetPositions(ctxPos);
                int xc = step.HiddenInput(carry, (long)_targets.Length * h, cn);
                for (int l = 0; l < _layers; l++)
                {
                    string cp = $"{Prefix}blk.{l}.";
                    int g = Mix(step, xc, cn, l, cp + "dflare_norm");
                    int ckv = step.MatMul(cp + "attn_kv_ctx", g);                                                    // [2 kv, cn]
                    int ck = step.Reshape(step.Cont(step.View2(ckv, kw, cn, 2 * kw * f, 0)), _headDim, _kvHeads, cn);
                    int cv = step.Reshape(step.Cont(step.View2(ckv, kw, cn, 2 * kw * f, kw * f)), _headDim, _kvHeads, cn);
                    ck = step.Rope(step.RmsNorm(ck, step.Weight(cp + "attn_k_ctx_norm"), _eps), _rotary);
                    for (int j = 0; j < m; j++)
                    {
                        int kj = step.Cont(step.View3(ck, _headDim, _kvHeads, cr, _headDim * f, kw * f, (long)j * cr * kw * f));
                        int vj = step.Cont(step.View3(cv, _headDim, _kvHeads, cr, _headDim * f, kw * f, (long)j * cr * kw * f));
                        var lanePos = new int[cr];
                        Array.Copy(ctxPos, j * cr, lanePos, 0, cr);
                        Write(step, cs[j], l, kj, vj, cr, lanePos);
                    }
                }
                // The carries (row cr j + b of each lane) for the Markov head.
                var carryRows = new int[m];
                for (int j = 0; j < m; j++) carryRows[j] = j * cr + b;
                int x0 = step.Op(GraphOp.GetRows, xc, step.InputI32(carryRows, m));                                // [feat, m]

                // The blocks: lane j's rows [anchor, mask x (b - 1)] at p_j .. p_j + b - 1.
                var positions = new int[m * b];
                for (int j = 0; j < m; j++)
                    for (int i = 0; i < b; i++) positions[j * b + i] = ps[j] + i;
                step.SetPositions(positions);
                int emb = step.RmsNorm(step.HiddenInput(embedded, h, m), step.Weight("token_embd_norm.weight"), _m.Config.Eps);   // [h, m]
                int table = step.Op(GraphOp.Concat, stackalloc[] { emb, step.Reshape(step.Weight(Prefix + "mask_embed"), h, 1) },
                    stackalloc long[] { 1 });                                                                         // [h, m + 1]
                var pick = new int[m * b];
                for (int j = 0; j < m; j++)
                    for (int i = 0; i < b; i++) pick[j * b + i] = i == 0 ? j : m;
                int x = step.Op(GraphOp.GetRows, table, step.InputI32(pick, m * b));                                 // [h, m b]
                float scale = 1f / MathF.Sqrt(_headDim);
                for (int l = 0; l < _layers; l++)
                {
                    string pre = $"{Prefix}blk.{l}.";
                    int hn = step.RmsNorm(x, step.Weight(pre + "attn_norm"), _eps);
                    int qkv = step.MatMul(pre + "attn_qkv", hn);                                                     // [q + 2 kv, m b]
                    int q = step.Reshape(step.Cont(step.View2(qkv, qw, m * b, qkvw * f, 0)), _headDim, _heads, m * b);
                    int k = step.Reshape(step.Cont(step.View2(qkv, kw, m * b, qkvw * f, qw * f)), _headDim, _kvHeads, m * b);
                    int v = step.Reshape(step.Cont(step.View2(qkv, kw, m * b, qkvw * f, (qw + kw) * f)), _headDim, _kvHeads, m * b);
                    q = step.Rope(step.RmsNorm(q, step.Weight(pre + "attn_q_norm"), _eps), _rotary);
                    k = step.Rope(step.RmsNorm(k, step.Weight(pre + "attn_k_norm"), _eps), _rotary);
                    int attnAll = -1;
                    for (int j = 0; j < m; j++)
                    {
                        Context c = cs[j];
                        int p = ps[j];
                        int kj = step.Cont(step.View3(k, _headDim, _kvHeads, b, _headDim * f, kw * f, j * b * kw * f));
                        int vj = step.Cont(step.View3(v, _headDim, _kvHeads, b, _headDim * f, kw * f, j * b * kw * f));
                        var lanePos = new int[b];
                        for (int i = 0; i < b; i++) lanePos[i] = p + i;
                        int wv = -1;
                        int wk = Write(step, c, l, kj, vj, b, lanePos, node => wv = node);
                        int keys, mask;
                        if (_local[l])
                        {
                            keys = _ringRows;
                            mask = LocalMask(step, p);
                        }
                        else
                        {
                            keys = Math.Min(c.Capacity, (p + b + 1023) / 1024 * 1024);
                            mask = GlobalMask(step, p, keys);
                        }
                        long rowBytes = (long)_headDim * 2;
                        int kv = step.View3(wk, _headDim, keys, _kvHeads, rowBytes, Rows(c, l) * rowBytes, 0);
                        int vv = step.View3(wv, _headDim, keys, _kvHeads, rowBytes, Rows(c, l) * rowBytes, 0);
                        int qj = step.View3(q, _headDim, _heads, b, _headDim * f, qw * f, j * b * qw * f);
                        int fa = step.FlashAttention(step.Permute(qj, 0, 2, 1, 3), kv, vv, mask, scale);
                        int laneAttn = step.Reshape(fa, qw, b);
                        attnAll = attnAll < 0 ? laneAttn : step.Op(GraphOp.Concat, stackalloc[] { attnAll, laneAttn }, stackalloc long[] { 1 });
                    }
                    int attn = step.MatMul(pre + "attn_output", attnAll);
                    x = step.Op(GraphOp.Add, attn, x);
                    int fn = step.RmsNorm(x, step.Weight(pre + "ffn_norm"), _eps);
                    int gu = step.MatMul(pre + "ffn_gate_up", fn);
                    int act = step.Op(GraphOp.Swiglu, gu);
                    x = step.Op(GraphOp.Add, step.MatMul(pre + "ffn_down", act), x);
                }
                int final = step.RmsNorm(x, step.Weight(Prefix + "output_norm"), _eps);
                // Each lane's mask rows, lane-major: row j gamma + i is lane j's mask i.
                var maskRows = new int[m * gamma];
                var laneOf = new int[m * gamma];
                for (int j = 0; j < m; j++)
                    for (int i = 0; i < gamma; i++) { maskRows[j * gamma + i] = j * b + 1 + i; laneOf[j * gamma + i] = j; }
                int rows = step.Op(GraphOp.GetRows, final, step.InputI32(maskRows, m * gamma));                      // [h, m gamma]
                int baseLogits = step.MatMul("output.weight", rows);                                                  // [vocab padded, m gamma]

                // The Markov head: each lane's context from its own carry, the walk over all lanes at once.
                int frnn = Mix(step, x0, m, _layers, Prefix + "markov.dflare_norm");                                  // [h, m]
                int frnnRows = step.Op(GraphOp.GetRows, frnn, step.InputI32(laneOf, m * gamma));                      // [h, m gamma]
                int ctxIn = step.Op(GraphOp.Concat, stackalloc[] { rows, frnnRows }, stackalloc long[] { 0 });           // [2 h, m gamma]
                int ctx = step.Op(GraphOp.Add, step.MatMul(Prefix + "markov.rnn_ctx", ctxIn), step.Weight(Prefix + "markov.rnn_bias"));
                int r = _rank;
                long padded = _m._vocabPadded;
                int state = step.ConstantInputF32(new float[(long)r * m], r, m);
                int prev = step.InputI32(anchors, m);
                int ids = TokenIds(step);
                var idOut = new int[gamma];
                var confOut = new int[gamma];
                for (int i = 0; i < gamma; i++)
                {
                    int w1 = step.Op(GraphOp.GetRows, step.Weight(Prefix + "markov.w1"), prev);                       // [r, m]
                    int sin = step.Op(GraphOp.Concat, stackalloc[] { state, w1 }, stackalloc long[] { 0 });             // [2 r, m]
                    int ctxI = step.Cont(step.View2(ctx, 3L * r, m, gamma * 3L * r * f, i * 3L * r * f));              // lane j's row j gamma + i
                    int raw = step.Op(GraphOp.Add, step.MatMul(Prefix + "markov.rnn_state", sin), ctxI);                // [3 r, m]
                    int gate = step.Op(GraphOp.Sigmoid, step.Cont(step.View2(raw, r, m, 3L * r * f, 0)));
                    int cand = step.Op(GraphOp.Tanh, step.Cont(step.View2(raw, r, m, 3L * r * f, r * f)));
                    int latent = step.Op(GraphOp.Tanh, step.Cont(step.View2(raw, r, m, 3L * r * f, 2L * r * f)));
                    int keep = step.Op(GraphOp.Mul, gate, state);
                    int fresh = step.Op(GraphOp.Mul, step.Op(GraphOp.ScaleBias, stackalloc[] { gate }, default, stackalloc[] { -1f, 1f }), cand);
                    state = step.Op(GraphOp.Add, keep, fresh);
                    int foldY = step.MatMul(Prefix + "markov.fold", latent);                                           // [padded, m]
                    int foldB = step.Weight(Prefix + "markov.fold_bias");
                    int baseV = step.View2(baseLogits, padded, m, gamma * padded * f, i * padded * f);
                    step.BeginFused(GraphFusion.InklingDraftPick, stackalloc[] { baseV, foldY, foldB }, stackalloc long[] { vocab },
                        stackalloc[] { _m._softcap });
                    int bias = step.Op(GraphOp.Add, foldY, foldB);
                    int baseI = step.Cont(baseV);
                    int logits = step.Op(GraphOp.Add, baseI, bias);                                                    // [padded, m]
                    int real = step.Cont(step.View2(logits, vocab, m, padded * f, 0));                                 // [vocab, m]
                    int best = step.Op(GraphOp.Argmax, real);                                                          // I32 [m]
                    int capped = step.Op(GraphOp.Scale, stackalloc[] { step.Op(GraphOp.Tanh, step.Op(GraphOp.Scale, stackalloc[] { real }, default,
                        stackalloc[] { 1f / _m._softcap })) }, default, stackalloc[] { _m._softcap });
                    int probs = step.Op(GraphOp.SoftMaxExt, stackalloc[] { capped }, default, stackalloc[] { 1f, 0f });
                    int conf = step.Op(GraphOp.GetRows, step.Reshape(probs, 1, vocab, m), step.Reshape(best, 1, m));
                    int idf = step.Op(GraphOp.GetRows, ids, best);
                    step.EndFused(stackalloc[] { best, conf, idf });
                    confOut[i] = step.Output(conf, m);
                    idOut[i] = step.Output(idf, m);
                    prev = best;
                }
                var draft = new int[2 * gamma];
                idOut.CopyTo(draft, 0);
                confOut.CopyTo(draft, gamma);
                return new StandardDecoderModel.StepOutputs { Logits = -1, Final = -1, Hidden = -1, Draft = draft };
            }

            /// <summary>The ids 0 .. vocab - 1 as F32 [1, vocab]: a pick is gathered from here as a float output.</summary>
            private int TokenIds(GraphStep step)
            {
                int vocab = _m.Config.VocabSize;
                if (_tokenIds == null)
                {
                    var ids = new float[vocab];
                    for (int i = 0; i < vocab; i++) ids[i] = i;
                    _tokenIds = _m.CreateFloatTensor(ids, vocab, 1);
                }
                return step.TensorWeight(_tokenIds, 1, vocab);
            }

            private const ushort NegInf = 0xFC00;

            /// <summary>A local layer's mask [ring rows, block] (F16): a cell is seen when it holds a position in
            /// [p - window, p + block) (the context's window and the block); its position is the last one written to it.</summary>
            private int LocalMask(GraphStep step, int p)
            {
                int rows = _ringRows, b = _block, last = p + b - 1;
                var mask = new ushort[(long)rows * b];
                for (int cell = 0; cell < rows; cell++)
                {
                    int pos = cell <= last ? cell + (last - cell) / rows * rows : -1;
                    ushort value = pos >= 0 && pos >= p - _window ? (ushort)0 : NegInf;
                    for (int i = 0; i < b; i++) mask[(long)i * rows + cell] = value;
                }
                return step.InputF16(mask, rows, b);
            }

            /// <summary>The global layer's mask [keys, block] (F16): positions before the block's end.</summary>
            private int GlobalMask(GraphStep step, int p, int keys)
            {
                int b = _block, end = p + b;
                var mask = new ushort[(long)keys * b];
                for (int i = 0; i < b; i++)
                    for (int key = end; key < keys; key++) mask[(long)i * keys + key] = NegInf;
                return step.InputF16(mask, keys, b);
            }
        }

        /// <summary>The DSpark drafter's tensors under "dspark.": block-quantized ones as quantized weights, the rest as F32.</summary>
        private void LoadDSparkTensors(GgufFile draft)
        {
            foreach (var kv in draft.Tensors)
            {
                var info = kv.Value;
                string name = info.Name;
                long byteCount = draft.GetTensorByteCount(info);
                if (info.Type == GgmlTensorType.Q8_0)
                {
                    if (IsGgmlBackend) EnsureQuantBackendAvailable();
                    IntPtr ptr = QuantizedWeight.AllocateBuffer(byteCount);
                    draft.ReadTensorDataToNative(info, ptr, byteCount);
                    _quantWeights[name] = new QuantizedWeight(ptr, byteCount, (int)info.Type, (long)info.Shape[0], (long)info.Shape[1]);
                    continue;
                }
                long[] shape = new long[info.Shape.Length];
                for (int i = 0; i < info.Shape.Length; i++) shape[i] = (long)info.Shape[info.Shape.Length - 1 - i];
                var tensor = new Tensor(_allocator, DType.Float32, shape);
                IntPtr dest = TensorComputePrimitives.GetStoragePointer(tensor);
                if (info.Type == GgmlTensorType.F32)
                    draft.ReadTensorDataToFloat32Native(info, dest, info.NumElements);
                else
                {
                    IntPtr temp = QuantizedWeight.AllocateBuffer(byteCount);
                    try
                    {
                        draft.ReadTensorDataToNative(info, temp, byteCount);
                        ManagedQuantizedOps.DequantizeToFloat32Native((int)info.Type, temp, dest, info.NumElements);
                    }
                    finally
                    {
                        QuantizedWeight.FreeBuffer(temp);
                    }
                }
                _weights[name] = tensor;
            }
        }

        /// <summary>A block drafter's captured row: the stream's current slot entering a run, or the head's input past the last.</summary>
        protected override int EmitCaptureRow(GraphStep step, int stream, int layer)
        {
            if (layer >= Config.NumLayers) return EmitStreamOutput(step, stream);
            // The value the previous run's FFN wrote (a pass's first run restarts the slot: not a drafter's target).
            if (ReferenceEquals(step, _curStep) && _curLayer == layer - 1 && !StartsPass(layer)) return _curValue;
            return StreamSlot(step, stream, Cur);
        }
    }
}
