// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Direct-backend MiniMax-H3 diffusion transformer: the same graph as
// TSGgml_MiniMaxH3DitForward, expressed against the shared DirectOps/DirectLinear
// primitives so it runs with no ggml at all. This is what makes BackendType.Cpu
// (the 100% pure-C# backend) reach MiniMax-H3, which was otherwise the one
// generation family whose every stage was an unconditional whole-model native
// call.
//
// Sequence order and packing are fixed by the caller: text, conditioning frames,
// target audio, target video. There is no attention mask anywhere in this model,
// and modulation is applied per segment of that packed sequence.
using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;

using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.GGML;          // H3DitSegment / H3CondChunk: plain layout structs, no native call
using TensorSharp.Models.Direct;
using TensorSharp.Runtime;

namespace TensorSharp.Models.MiniMaxH3
{
    /// <summary>Managed MiniMax-H3 DiT step (no ggml).</summary>
    internal sealed class MiniMaxH3DirectDiT : IDisposable
    {
        private sealed class BlockW
        {
            public Tensor Norm1, Norm2, QNorm, KNorm;
            public DirectLinear AdaLn, Qkv, Out, Fc1, Fc2;
        }

        private readonly DirectContext _ctx;
        private readonly MiniMaxH3TensorOps _tensorOps;
        private readonly MiniMaxH3Config _cfg;
        private readonly BlockW[] _blocks;
        private readonly BlockW[] _refiner;      // no AdaLn on refiner blocks
        private readonly DirectLinear _videoPatch, _audioPatch, _conditionProj;
        private readonly DirectLinear _finalAdaLn, _finalVideoOut, _finalAudioOut;
        private readonly Tensor _finalNorm, _refinerFinalNorm;
        private readonly List<DirectLinear> _owned = new();
        // The projection and refiner depend only on text conditioning, which is
        // constant throughout denoising. Keep one prompt, on the selected device.
        // Snapshot its bytes because callers may reuse and mutate their buffer.
        private float[] _cachedTextHidden;
        private Tensor _cachedTextTokens;
        // The refiner blocks carry LoRA updates too, so the cached tokens belong to the
        // plug-in they were refined with as much as to the prompt.
        private MiniMaxH3LoraSet _cachedTextLora;
        // The LoRA plug-in applied unmerged (a direct-mode MiniMaxH3LoraSet), null for none.
        private LoraState _lora;
        private bool _disposed;

        /// <summary>
        /// One block's LoRA terms as exact F32 linears: down [rank, in], and up [out, rank]
        /// with strength * alpha / rank folded in and fc1's halves already in [gate | value]
        /// order (MiniMaxH3LoraSet.ReadFactors). q, k and v read one input, so their downs are
        /// stacked into one projection whose output each part's up reads at its own offset.
        /// </summary>
        private sealed class BlockLora
        {
            public DirectLinear QkvDown;
            public readonly DirectLinear[] QkvUp = new DirectLinear[3];
            public readonly int[] QkvOffset = new int[3], QkvRank = new int[3];
            public DirectLinear OutDown, OutUp, Fc1Down, Fc1Up, Fc2Down, Fc2Up;
        }

        /// <summary>A block the plug-in leaves alone, so it is not looked up again.</summary>
        private static readonly BlockLora NoBlockLora = new();

        /// <summary>
        /// The factors of one plug-in, in a context of their own so a set change or a residency
        /// release drops them without touching the checkpoint's weights. Blocks are built the
        /// first time a forward reaches them - inside <see cref="Forward"/>, so on MLX the arrays
        /// are created on its worker - and kept for the steps that follow.
        /// </summary>
        private sealed class LoraState : IDisposable
        {
            public readonly MiniMaxH3LoraSet Set;
            public readonly DirectContext Context;
            public readonly BlockLora[] Trunk, Refiner;

            public LoraState(MiniMaxH3LoraSet set, IAllocator allocator, int blocks, int refinerBlocks)
            {
                Set = set;
                Context = new DirectContext(allocator);
                Trunk = new BlockLora[blocks];
                Refiner = new BlockLora[refinerBlocks];
            }

            public void Dispose() => Context.Dispose();
        }

        // Explicit diagnostic boundary snapshots. Device synchronization and
        // downloads occur only when this evidence directory is requested.
        private static readonly string TraceDirectory =
            Environment.GetEnvironmentVariable("TS_H3_DIT_TRACE_DIR");

        private static void Trace(string name, float[] values)
        {
            if (string.IsNullOrEmpty(TraceDirectory)) return;
            Directory.CreateDirectory(TraceDirectory);
            using var stream = File.Create(Path.Combine(TraceDirectory, name + ".f32"));
            using var writer = new BinaryWriter(stream);
            foreach (float value in values) writer.Write(value);
        }

        private static void Trace(string name, Tensor value)
        {
            if (!string.IsNullOrEmpty(TraceDirectory)) Trace(name, DirectOps.ToArray(value));
        }

        public MiniMaxH3DirectDiT(GgufFile gguf, MiniMaxH3Config config, IAllocator allocator)
        {
            _cfg = config;
            _ctx = new DirectContext(allocator);
            _tensorOps = new MiniMaxH3TensorOps(_ctx);

            string p = config.Prefix;
            DirectLinear Lin(string w, string b)
            {
                var l = DirectLinear.FromGguf(_ctx, gguf, w, b, prescale: true, h3PreserveFloat32: true);
                _owned.Add(l);
                return l;
            }
            Tensor Vec(string name) =>
                _ctx.Own(_ctx.FromFloats(DirectOps.DequantTensor(gguf, name),
                                         (long)gguf.Tensors[name].NumElements));

            _videoPatch = Lin(p + "video_patch_proj.weight", p + "video_patch_proj.bias");
            _audioPatch = Lin(p + "audio_patch_proj.weight", p + "audio_patch_proj.bias");
            _conditionProj = Lin(p + "condition_proj.weight", p + "condition_proj.bias");
            _finalAdaLn = Lin(p + "final_layer.adaln_proj.linear.weight",
                              p + "final_layer.adaln_proj.linear.bias");
            _finalVideoOut = Lin(p + "final_layer.video_out.weight", p + "final_layer.video_out.bias");
            _finalAudioOut = Lin(p + "final_layer.audio_out.weight", p + "final_layer.audio_out.bias");
            _finalNorm = Vec(p + "final_layer.norm.weight");
            _refinerFinalNorm = Vec(p + "token_refiner.final_norm.weight");

            _refiner = new BlockW[config.TokenRefinerLayers];
            for (int i = 0; i < _refiner.Length; i++)
            {
                string b = p + "token_refiner.blocks." + i + ".";
                _refiner[i] = new BlockW
                {
                    Norm1 = Vec(b + "norm1.weight"),
                    Norm2 = Vec(b + "norm2.weight"),
                    QNorm = Vec(b + "attn.q_norm.weight"),
                    KNorm = Vec(b + "attn.k_norm.weight"),
                    Qkv = Lin(b + "attn.qkv_proj.weight", null),
                    Out = Lin(b + "attn.out_proj.weight", null),
                    Fc1 = Lin(b + "mlp.fc1.weight", null),
                    Fc2 = Lin(b + "mlp.fc2.weight", null),
                };
            }

            _blocks = new BlockW[config.NumLayers];
            for (int i = 0; i < _blocks.Length; i++)
            {
                string b = p + "blocks." + i + ".";
                _blocks[i] = new BlockW
                {
                    Norm1 = Vec(b + "norm1.weight"),
                    Norm2 = Vec(b + "norm2.weight"),
                    QNorm = Vec(b + "attn.q_norm.weight"),
                    KNorm = Vec(b + "attn.k_norm.weight"),
                    AdaLn = Lin(b + "adaln_proj.linear.weight", b + "adaln_proj.linear.bias"),
                    Qkv = Lin(b + "attn.qkv_proj.weight", null),
                    Out = Lin(b + "attn.out_proj.weight", null),
                    Fc1 = Lin(b + "mlp.fc1.weight", null),
                    Fc2 = Lin(b + "mlp.fc2.weight", null),
                };
            }
        }

        // ---- small helpers ---------------------------------------------------

        private static Tensor Rows(Tensor src, int from, int count)
        {
            using var view = src.Narrow(0, from, count);
            // Full rows of a contiguous matrix already have the required
            // layout. Retain their storage until packing/norm finishes instead
            // of allocating and copying another device buffer.
            return Ops.AsContiguous(view);
        }

        private static Tensor Cols(Tensor src, int from, int count)
        {
            using var view = src.Narrow(1, from, count);
            return Ops.NewContiguous(view);
        }

        /// <summary>One row of a modulation table as a contiguous [cols] vector.</summary>
        private static Tensor ModRow(Tensor mods, int row)
        {
            using var view = mods.Narrow(0, row, 1);
            using var c = Ops.AsContiguous(view);
            return c.View(mods.Sizes[1]);
        }

        private static void CopyRows(Tensor dst, int dstRow, Tensor src)
        {
            using var target = dst.Narrow(0, dstRow, src.Sizes[0]);
            Ops.Copy(target, src);
        }

        /// <summary>
        /// Rotate-half RoPE over the leading <paramref name="rot"/> channels of every
        /// head, in place. cos/sin are per-token tables [seq, rot] over the packed
        /// sequence - NOT the pair-duplicated layout DirectOps.RopeInterleaved wants,
        /// which is why this model needs its own.
        /// </summary>
        private unsafe void RopeHalf(Tensor x, float[] cos, float[] sin,
                                            int seq, int heads, int hd, int rot)
        {
            if (rot <= 0) return;
            if (_tensorOps.IsAccelerated)
            {
                _tensorOps.RopeHalf(x, cos, sin, seq, heads, hd, rot);
                return;
            }
            int half = rot / 2;
            float* px = (float*)CpuNativeHelpers.GetBufferStart(x);
            fixed (float* pcos = cos, psin = sin)
            {
                float* pc = pcos, ps = psin;
                CpuWorkerPool.Shared.For(seq, t =>
                {
                    float* ct = pc + (long)t * rot;
                    float* st = ps + (long)t * rot;
                    float* row = px + (long)t * heads * hd;
                    for (int h = 0; h < heads; h++)
                    {
                        float* v = row + (long)h * hd;
                        for (int i = 0; i < half; i++)
                        {
                            // rotate_half(x) = [-x2, x1]
                            float a = v[i], b = v[i + half];
                            v[i] = a * ct[i] - b * st[i];
                            v[i + half] = b * ct[i + half] + a * st[i + half];
                        }
                    }
                });
            }
        }

        private Tensor Attention(Tensor h, BlockW w, int seq, float[] cos, float[] sin, bool applyRope = true,
                                 BlockLora lora = null)
        {
            int heads = _cfg.NumAttentionHeads, hd = _cfg.AttentionHeadDim;
            int inner = _cfg.AttentionInnerDim, rot = _cfg.RopeRotaryDim;
            float scale = 1.0f / MathF.Sqrt(hd);

            Tensor q, k;
            Tensor v;
            using (Tensor qkv = w.Qkv.Forward(h))          // [seq, 3*inner]
            {
                q = Cols(qkv, 0, inner);
                k = Cols(qkv, inner, inner);
                v = Cols(qkv, 2 * inner, inner);
            }

            try
            {
                // LoRA updates land on q/k/v before the q/k norm and RoPE, where the
                // reference's to_q/to_k/to_v adapters put them (transformer_minimax_h3.py).
                if (lora?.QkvDown != null) AddQkvLora(lora, h, q, k, v);

                // q/k RMS norm is per HEAD, over the head dim.
                if (w.QNorm != null) q = NormHeads(q, w.QNorm, seq, heads, hd);
                if (w.KNorm != null) k = NormHeads(k, w.KNorm, seq, heads, hd);

                if (applyRope)
                {
                    RopeHalf(q, cos, sin, seq, heads, hd, rot);
                    RopeHalf(k, cos, sin, seq, heads, hd, rot);
                }

                using Tensor merged = DirectOps.Attention(_ctx, q, k, v, heads, hd, scale);
                Tensor o = w.Out.Forward(merged);
                if (lora?.OutDown == null) return o;
                try
                {
                    AddLora(lora.OutDown, lora.OutUp, merged, o);
                    return o;
                }
                catch { o.Dispose(); throw; }
            }
            finally
            {
                q?.Dispose();
                k?.Dispose();
                v?.Dispose();
            }
        }

        private Tensor NormHeads(Tensor x, Tensor gain, int seq, int heads, int hd)
        {
            using (x)
            using (Tensor flat = x.View((long)seq * heads, hd))
            using (Tensor n = DirectOps.RmsNorm(_ctx, flat, gain, _cfg.NormEps))
                return n.View(seq, (long)heads * hd);
        }

        /// <summary>SwiGLU with the gate first: fc1 emits [gate | value].</summary>
        private Tensor Mlp(Tensor h, BlockW w, BlockLora lora = null)
        {
            int ffn = _cfg.FfnHiddenSize;
            using Tensor gv = w.Fc1.Forward(h);
            // fc1's update is already in [gate | value] order, so it joins gv before the SwiGLU.
            if (lora?.Fc1Down != null) AddLora(lora.Fc1Down, lora.Fc1Up, h, gv);
            using Tensor act = ActivateMlp(_ctx, gv, ffn);
            Tensor y = w.Fc2.Forward(act);
            if (lora?.Fc2Down == null) return y;
            try
            {
                AddLora(lora.Fc2Down, lora.Fc2Up, act, y);
                return y;
            }
            catch { y.Dispose(); throw; }
        }

        // ---- LoRA plug-in ----------------------------------------------------

        /// <summary>y += up(down(x)), in place: the unmerged update of one projection.</summary>
        private static void AddLora(DirectLinear down, DirectLinear up, Tensor x, Tensor y)
        {
            using Tensor low = down.Forward(x);
            using Tensor delta = up.Forward(low);
            Ops.Add(y, y, delta);
        }

        /// <summary>q/k/v += their updates: one stacked shrink of <paramref name="h"/>, then
        /// each part's up over its own columns of it.</summary>
        private static void AddQkvLora(BlockLora lora, Tensor h, Tensor q, Tensor k, Tensor v)
        {
            using Tensor low = lora.QkvDown.Forward(h);   // [seq, sum of the q/k/v ranks]
            Tensor[] parts = { q, k, v };
            for (int i = 0; i < 3; i++)
            {
                if (lora.QkvUp[i] == null) continue;
                using Tensor slice = Cols(low, lora.QkvOffset[i], lora.QkvRank[i]);
                using Tensor delta = lora.QkvUp[i].Forward(slice);
                Ops.Add(parts[i], parts[i], delta);
            }
        }

        /// <summary>Apply <paramref name="lora"/> from the next block on. A different set, or
        /// none, drops the previous one's factors and the text it refined.</summary>
        private void UseLora(MiniMaxH3LoraSet lora)
        {
            if (ReferenceEquals(_lora?.Set, lora)) return;
            ReleaseLora();
            ClearTextCache();
            if (lora != null) _lora = new LoraState(lora, _ctx.Allocator, _blocks.Length, _refiner.Length);
        }

        /// <summary>Drop the plug-in's built factors; a later forward rebuilds them from the set.</summary>
        internal void ReleaseLora()
        {
            _lora?.Dispose();
            _lora = null;
        }

        /// <summary>The block's terms, built on first use; null when the plug-in has none there.</summary>
        private BlockLora LoraFor(bool refiner, int block)
        {
            if (_lora == null) return null;
            BlockLora[] built = refiner ? _lora.Refiner : _lora.Trunk;
            built[block] ??= BuildBlockLora(refiner, block);
            return ReferenceEquals(built[block], NoBlockLora) ? null : built[block];
        }

        private BlockLora BuildBlockLora(bool refiner, int block)
        {
            MiniMaxH3LoraSet set = _lora.Set;
            DirectContext ctx = _lora.Context;
            // down [rank, in] is ggml [ne0 = in, ne1 = rank]; up [out, rank] is [ne0 = rank, ne1 = out].
            // The factors never change once built, so on MLX they go straight into device arrays
            // with no host copy kept beside them (that copy would be another 2.77 GB for the 8-step adapter).
            DirectLinear Linear(float[] w, long input, long output) =>
                DirectLinear.FromFloats(ctx, w, input, output, null, exactFloat32: true, mlxDeviceOnly: ctx.IsMlx);

            var b = new BlockLora();
            bool any = false;
            var qkv = new (float[] Down, float[] Up, int Rank, long In, long Out)?[3];
            int stacked = 0;
            long width = 0;
            for (int i = 0; i < 3; i++)
            {
                qkv[i] = set.ReadFactors(refiner, block, (MiniMaxH3LoraTarget)i);
                if (qkv[i] is not { } f) continue;
                b.QkvOffset[i] = stacked;
                b.QkvRank[i] = f.Rank;
                stacked += f.Rank;
                width = f.In;
            }
            if (stacked > 0)
            {
                var down = new float[(long)stacked * width];
                for (int i = 0; i < 3; i++)
                {
                    if (qkv[i] is not { } f) continue;
                    Array.Copy(f.Down, 0, down, (long)b.QkvOffset[i] * width, (long)f.Rank * width);
                    b.QkvUp[i] = Linear(f.Up, f.Rank, f.Out);
                }
                b.QkvDown = Linear(down, width, stacked);
                any = true;
            }
            (DirectLinear Down, DirectLinear Up) Pair(MiniMaxH3LoraTarget t)
            {
                if (set.ReadFactors(refiner, block, t) is not { } f) return (null, null);
                any = true;
                return (Linear(f.Down, f.In, f.Rank), Linear(f.Up, f.Rank, f.Out));
            }
            (b.OutDown, b.OutUp) = Pair(MiniMaxH3LoraTarget.Out);
            (b.Fc1Down, b.Fc1Up) = Pair(MiniMaxH3LoraTarget.Fc1);
            (b.Fc2Down, b.Fc2Up) = Pair(MiniMaxH3LoraTarget.Fc2);
            return any ? b : NoBlockLora;
        }

        internal static Tensor ActivateMlp(DirectContext context, Tensor gv, int ffn, bool fuseAccelerated = true)
        {
            // The device split kernel reads [gate | value] directly and keeps
            // the same SiLU-then-multiply arithmetic. Avoid two full activation
            // copies and the separate multiply pass over the large FFN width.
            if (fuseAccelerated && context.IsAccelerated)
                return Ops.SiLUMulSplit(null, gv, ffn);

            using Tensor gate = Cols(gv, 0, ffn);
            using Tensor val = Cols(gv, ffn, ffn);
            Tensor act = DirectOps.Silu(context, gate);
            try
            {
                Ops.Mul(act, act, val);
                return act;
            }
            catch { act.Dispose(); throw; }
        }

        // ---- modulation over the packed sequence -----------------------------

        /// <summary>x = x * (1 + scale) + shift, one modulation row per segment.</summary>
        private void Modulate(Tensor h, Tensor mods, H3DitSegment[] segs, int shiftIdx, int scaleIdx)
        {
            if (_tensorOps.TryModulatePacked(h, null, mods, segs, shiftIdx, scaleIdx, false)) return;
            foreach (var sg in segs)
            {
                int n = sg.End - sg.Start;
                if (n <= 0) continue;
                using Tensor sh = ModRow(mods, sg.Col + shiftIdx);
                using Tensor sc = ModRow(mods, sg.Col + scaleIdx);
                DirectOps.ModulateRows(_ctx, h, h, sh, sc, sg.Start, n);
            }
        }

        /// <summary>x += upd * gate, one gate row per segment.</summary>
        private void GatedResidual(Tensor x, Tensor upd, Tensor mods, H3DitSegment[] segs, int gateIdx)
        {
            if (_tensorOps.TryModulatePacked(x, upd, mods, segs, gateIdx, gateIdx, true)) return;
            foreach (var sg in segs)
            {
                int n = sg.End - sg.Start;
                if (n <= 0) continue;
                using Tensor g = ModRow(mods, sg.Col + gateIdx);
                DirectOps.GateAddRows(_ctx, x, upd, g, sg.Start, n);
            }
        }

        private void ClearTextCache()
        {
            _cachedTextTokens?.Dispose();
            _cachedTextTokens = null;
            _cachedTextHidden = null;
            _cachedTextLora = null;
        }

        // The returned reference is borrowed storage with its own lifetime. The
        // caller copies it into the packed sequence before any trunk mutation.
        internal Tensor RefineText(float[] textHidden, int textCount)
        {
            if (textHidden == null || textCount <= 0 ||
                textHidden.LongLength != (long)textCount * _cfg.TextDim)
                throw new ArgumentException("MiniMax-H3 text conditioning has an invalid shape.");

            // Trace mode deliberately executes every boundary, including the
            // intermediate refiner blocks that are absent from the final cache.
            bool cache = string.IsNullOrEmpty(TraceDirectory);
            if (cache && _cachedTextTokens != null && ReferenceEquals(_cachedTextLora, _lora?.Set) &&
                MemoryMarshal.AsBytes(textHidden.AsSpan()).SequenceEqual(
                    MemoryMarshal.AsBytes(_cachedTextHidden.AsSpan())))
                return _cachedTextTokens.CopyRef();

            ClearTextCache();
            Tensor ctxTok = null;
            try
            {
                using (Tensor tin = _ctx.FromFloats(textHidden, textCount, _cfg.TextDim))
                    ctxTok = _conditionProj.Forward(tin);
                Trace("condition-proj", ctxTok);

                // The official H3 token refiner has no positional embedding.
                // RoPE belongs to the subsequent packed transformer trunk.
                int refinerIndex = 0;
                foreach (var rw in _refiner)
                {
                    BlockLora rl = LoraFor(refiner: true, refinerIndex);
                    using (Tensor n1 = DirectOps.RmsNorm(_ctx, ctxTok, rw.Norm1, _cfg.NormEps))
                    using (Tensor a = Attention(n1, rw, textCount, null, null, applyRope: false, rl))
                        Ops.Add(ctxTok, ctxTok, a);
                    using (Tensor n2 = DirectOps.RmsNorm(_ctx, ctxTok, rw.Norm2, _cfg.NormEps))
                    using (Tensor m = Mlp(n2, rw, rl))
                        Ops.Add(ctxTok, ctxTok, m);
                    Trace("refiner-" + refinerIndex++, ctxTok);
                }
                if (_refinerFinalNorm != null)
                {
                    using Tensor prev = ctxTok;
                    ctxTok = DirectOps.RmsNorm(_ctx, prev, _refinerFinalNorm, _cfg.NormEps);
                }

                if (!cache)
                {
                    Tensor result = ctxTok;
                    ctxTok = null;
                    return result;
                }
                _cachedTextHidden = (float[])textHidden.Clone();
                _cachedTextLora = _lora?.Set;
                _cachedTextTokens = ctxTok;
                ctxTok = null;
                return _cachedTextTokens.CopyRef();
            }
            finally { ctxTok?.Dispose(); }
        }

        // ---- forward ---------------------------------------------------------

        /// <summary>
        /// One diffusion step over the packed sequence. The inputs are exactly the
        /// buffers the native descriptor is handed, so the two paths can be
        /// compared step for step. <paramref name="lora"/> (a direct-mode set, or null)
        /// adds its unmerged updates to the trunk and token-refiner blocks, as the
        /// native graph does with the set's packed factors.
        /// </summary>
        public (float[] Video, float[] Audio) Forward(
            float[] videoInput, int conditionCount, int videoCount,
            float[] audioLatent, int audioCount,
            float[] conditionAudio, int conditionAudioCount,
            IReadOnlyList<H3CondChunk> conditionChunks,
            float[] textHidden, int textCount,
            float[] timeEmbed, int nTimesteps,
            float[] cos, float[] sin,
            H3DitSegment[] segments,
            int nTok, int audioStart, int videoStart,
            int audioCol, int videoCol,
            float videoScale, float audioScale,
            int numBlocks, MiniMaxH3LoraSet lora = null)
        {
            int hidden = _cfg.HiddenSize;
            float eps = _cfg.NormEps;
            UseLora(lora);

            using Tensor temb = _ctx.FromFloats(timeEmbed, nTimesteps, _cfg.TimeEmbedDim);
            Trace("time-embed", timeEmbed);
            Trace("physical-audio-input", audioLatent);
            Trace("video-input", videoInput);
            if (textHidden != null) Trace("text-hidden-input", textHidden);

            // ---- text: project, then refine ----
            Tensor ctxTok = null;
            if (textCount > 0 && textHidden != null)
                ctxTok = RefineText(textHidden, textCount);
            else
                ClearTextCache();

            // ---- patchify video and audio, then pack ----
            // Conditioning frames share the patch projection of the target, exactly
            // as the reference concatenates them before projecting once.
            Tensor allVideo, audioTok, condAudioTok = null;
            using (Tensor vin = _ctx.FromFloats(videoInput, conditionCount + videoCount, _cfg.VideoPatchDim))
                allVideo = _videoPatch.Forward(vin);
            using (Tensor ain = _ctx.FromFloats(audioLatent, audioCount, _cfg.AudioLatentChannels))
                audioTok = _audioPatch.Forward(ain);
            if (conditionAudio != null && conditionAudioCount > 0)
            {
                using Tensor acin = _ctx.FromFloats(conditionAudio, conditionAudioCount,
                                                    _cfg.AudioLatentChannels);
                condAudioTok = _audioPatch.Forward(acin);
            }

            Tensor condTok = null, videoTok = allVideo;
            if (conditionCount > 0)
            {
                condTok = Rows(allVideo, 0, conditionCount);
                videoTok = Rows(allVideo, conditionCount, videoCount);
                allVideo.Dispose();
            }

            // With a chunk list the conditioning run is rebuilt in the order the
            // caller gave, taking each chunk from whichever projection it named.
            if (conditionChunks != null && conditionChunks.Count > 0)
            {
                var rebuilt = _ctx.NewF32(conditionCount + conditionAudioCount, hidden);
                int dst = 0, videoRow = 0, audioRow = 0;
                foreach (var ch in conditionChunks)
                {
                    if (ch.Count <= 0) continue;
                    if (ch.Kind == 0)
                    {
                        if (condTok == null || videoRow + ch.Count > conditionCount)
                            throw new InvalidOperationException(
                                "MiniMax-H3 DiT: conditioning video chunks overrun.");
                        using Tensor part = Rows(condTok, videoRow, ch.Count);
                        CopyRows(rebuilt, dst, part);
                        videoRow += ch.Count;
                    }
                    else
                    {
                        if (condAudioTok == null || audioRow + ch.Count > conditionAudioCount)
                            throw new InvalidOperationException(
                                "MiniMax-H3 DiT: conditioning audio chunks overrun.");
                        using Tensor part = Rows(condAudioTok, audioRow, ch.Count);
                        CopyRows(rebuilt, dst, part);
                        audioRow += ch.Count;
                    }
                    dst += ch.Count;
                }
                if (videoRow != conditionCount || audioRow != conditionAudioCount)
                    throw new InvalidOperationException(
                        "MiniMax-H3 DiT: conditioning chunks do not cover every row.");
                condTok?.Dispose();
                condTok = rebuilt;
            }
            condAudioTok?.Dispose();

            // Sequence order is fixed: text, conditioning frames, target audio, target video.
            Tensor x = _ctx.NewF32(nTok, hidden);
            int row = 0;
            void Append(Tensor part)
            {
                if (part == null) return;
                CopyRows(x, row, part);
                row += (int)part.Sizes[0];
                part.Dispose();
            }
            Append(ctxTok);
            Append(condTok);
            Append(audioTok);
            Append(videoTok);
            if (row != nTok)
                throw new InvalidOperationException(
                    $"MiniMax-H3 DiT: packed {row} tokens but the layout declares {nTok}.");
            Trace("packed-in", x);

            try
            {
                int nl = numBlocks > 0 ? Math.Min(numBlocks, _blocks.Length) : _blocks.Length;
                for (int l = 0; l < nl; l++)
                {
                    BlockW bw = _blocks[l];
                    BlockLora bl = LoraFor(refiner: false, l);
                    // No SiLU: the curve-table variant feeds the interpolated
                    // embedding straight into the projection.
                    using Tensor modsRaw = bw.AdaLn.Forward(temb);          // [nts, 18*hidden]
                    using Tensor mods = modsRaw.View(18L * nTimesteps, hidden);
                    if (l == 0) Trace("block-0-mods", mods);

                    using (Tensor h = DirectOps.RmsNorm(_ctx, x, bw.Norm1, eps))
                    {
                        Modulate(h, mods, segments, 0, 1);
                        using Tensor a = Attention(h, bw, nTok, cos, sin, lora: bl);
                        GatedResidual(x, a, mods, segments, 2);
                    }
                    using (Tensor h = DirectOps.RmsNorm(_ctx, x, bw.Norm2, eps))
                    {
                        Modulate(h, mods, segments, 3, 4);
                        using Tensor m = Mlp(h, bw, bl);
                        GatedResidual(x, m, mods, segments, 5);
                    }
                    Trace("block-" + l, x);
                }

                // ---- final layer ----
                // The final projection emits only shift+scale, and the spans index
                // it by TIMESTEP row rather than timestep*3+modality, so the column
                // bases differ from the per-block ones.
                using Tensor fmodsRaw = _finalAdaLn.Forward(temb);          // [nts, 2*hidden]
                using Tensor fmods = fmodsRaw.View(2L * nTimesteps, hidden);

                Tensor FinalSlice(int start, int count, int col)
                {
                    Tensor part;
                    using (Tensor slice = Rows(x, start, count))
                        part = DirectOps.RmsNorm(_ctx, slice, _finalNorm, eps);
                    using Tensor sh = ModRow(fmods, col);
                    using Tensor sc = ModRow(fmods, col + 1);
                    DirectOps.ModulateRows(_ctx, part, part, sh, sc, 0, count);
                    return part;
                }

                // The model emits velocities: video negated, audio scaled by
                // d(sigma_audio)/d(sigma_video) so one Euler step advances both.
                float[] videoOut, audioOut;
                using (Tensor vs = FinalSlice(videoStart, videoCount, videoCol))
                using (Tensor vo = _finalVideoOut.Forward(vs))
                {
                    Trace("raw-video-output", vo);
                    Ops.Mul(vo, vo, videoScale);
                    videoOut = DirectOps.ToArray(vo);
                }
                using (Tensor asl = FinalSlice(audioStart, audioCount, audioCol))
                using (Tensor ao = _finalAudioOut.Forward(asl))
                {
                    Trace("raw-audio-output", ao);
                    Ops.Mul(ao, ao, audioScale);
                    audioOut = DirectOps.ToArray(ao);
                }
                return (videoOut, audioOut);
            }
            finally
            {
                x.Dispose();
            }
        }

        internal void ReleaseDeviceResidency()
        {
            if (_disposed) return;
            ClearTextCache();
            // The plug-in's factors are the DiT's to rebuild, from the set, on the next forward.
            ReleaseLora();
            _ctx.ReleaseDeviceResidency();
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            ClearTextCache();
            ReleaseLora();
            foreach (var l in _owned) l.Dispose();
            _owned.Clear();
            _tensorOps.Dispose();
            _ctx.Dispose();
        }
    }
}
