// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;

namespace TensorSharp.Cuda
{
    public sealed unsafe partial class Q35CudaEngine
    {
        private const long ScratchAlignment = 256;

        /// <summary>Whether this engine shares storage between its mutually exclusive scratch phases.
        /// Selected once at construction by TS_Q35_SHARED_SCRATCH=1; disabled by default.</summary>
        public bool SharedScratchEnabled { get; private set; }
        /// <summary>The one owned backing allocation, or zero when sharing is disabled.</summary>
        public IntPtr SharedScratchBase { get; private set; }
        public long SharedScratchPoolBytes { get; private set; }
        /// <summary>Bytes avoided relative to the same ten separate Alloc calls (including their 256-byte minimum).</summary>
        public long SharedScratchBytesSaved { get; private set; }

        /// <summary>A diagnostic device span. Capacity includes padding; Bytes is the kernel-visible extent.
        /// Spans in one Phase are disjoint. Different phases may overlap only when sharing is enabled.</summary>
        public readonly record struct ScratchBufferSpan(string Name, string Phase, IntPtr Pointer, long Bytes, long CapacityBytes);
        public IReadOnlyList<ScratchBufferSpan> ScratchBufferSpans { get; private set; } = Array.Empty<ScratchBufferSpan>();

        internal readonly record struct ScratchLayoutSpan(string Name, string Phase, long Offset, long Bytes, long CapacityBytes);
        internal sealed record SharedScratchLayout(long PoolBytes, long LegacyBytes, IReadOnlyList<ScratchLayoutSpan> Spans)
        {
            public long BytesSaved => LegacyBytes - PoolBytes;
        }

        internal static bool SharedScratchRequested(string value) => value == "1";

        private static long AlignedScratchBytes(long bytes) => checked((Math.Max(ScratchAlignment, bytes) + ScratchAlignment - 1) & -ScratchAlignment);

        // Pure geometry: independently lay out normalize, attention+FFN, and GDN phases.
        // Keep every member of a phase disjoint even for zero/partial rows. Checked arithmetic
        // rejects impossible shapes before any allocation or pointer is made.
        internal static SharedScratchLayout PlanSharedScratch(int rows, int hiddenSize, int queryDim,
            int convDim, int valueHeads, int valueDim, int ffSize)
        {
            if (rows < 0) throw new ArgumentOutOfRangeException(nameof(rows));
            if (hiddenSize < 0) throw new ArgumentOutOfRangeException(nameof(hiddenSize));
            if (queryDim < 0) throw new ArgumentOutOfRangeException(nameof(queryDim));
            if (convDim < 0) throw new ArgumentOutOfRangeException(nameof(convDim));
            if (valueHeads < 0) throw new ArgumentOutOfRangeException(nameof(valueHeads));
            if (valueDim < 0) throw new ArgumentOutOfRangeException(nameof(valueDim));
            if (ffSize < 0) throw new ArgumentOutOfRangeException(nameof(ffSize));
            var spans = new List<ScratchLayoutSpan>(10);
            long legacy = 0;
            void Add(string name, string phase, int width, ref long offset)
            {
                long bytes = checked((long)rows * width * sizeof(float));
                long capacity = AlignedScratchBytes(bytes);
                spans.Add(new ScratchLayoutSpan(name, phase, offset, bytes, capacity));
                offset = checked(offset + capacity);
                legacy = checked(legacy + Math.Max(ScratchAlignment, bytes));
            }
            long norm = 0, attention = 0, gdn = 0;
            Add("_xn", "normalize", hiddenSize, ref norm);
            Add("_q", "attention", queryDim, ref attention);
            Add("_gate", "attention", queryDim, ref attention);
            Add("_attn", "attention", queryDim, ref attention);
            Add("_act", "attention", ffSize, ref attention);
            Add("_gdnQkv", "gdn", convDim, ref gdn);
            Add("_gdnG", "gdn", valueHeads, ref gdn);
            Add("_gdnBeta", "gdn", valueHeads, ref gdn);
            Add("_gdnO", "gdn", valueDim, ref gdn);
            Add("_gdnY", "gdn", valueDim, ref gdn);
            long pool = Math.Max(norm, Math.Max(attention, gdn));
            _ = checked((nint)pool);
            return new SharedScratchLayout(pool, legacy, spans.AsReadOnly());
        }

        internal static IntPtr SharedScratchPointer(IntPtr backing, long poolBytes, ScratchLayoutSpan span)
        {
            if (backing == IntPtr.Zero || (backing.ToInt64() & (ScratchAlignment - 1)) != 0)
                throw new ArgumentException("[q35-cuda] scratch backing must be nonzero and 256-byte aligned", nameof(backing));
            if (span.Offset < 0 || span.Offset % ScratchAlignment != 0 || span.Bytes < 0
                || span.CapacityBytes < Math.Max(ScratchAlignment, span.Bytes) || span.CapacityBytes % ScratchAlignment != 0
                || span.CapacityBytes > poolBytes || span.Offset > poolBytes - span.CapacityBytes)
                throw new ArgumentOutOfRangeException(nameof(span), "[q35-cuda] scratch span is outside its backing allocation");
            long address = checked(backing.ToInt64() + span.Offset);
            _ = checked((nint)checked(address + span.CapacityBytes - 1));
            return new IntPtr(address);
        }

        private void AllocateSharedScratch(int rows)
        {
            var layout = PlanSharedScratch(rows, _m.NEmbd, QDim, ConvDim, _m.GdnVHeads, ValueDim, _m.NFf);
            // InProj quantizes all _xn rows into independent Q81/FP4 storage before any
            // group writes attention/GDN scratch. OutProj consumes that phase before
            // post-norm writes _xn again; GateUp consumes _xn before SwiGLU writes _act.
            // Attention needs q/gate/attn AND _act (key-split partials) simultaneously.
            // MixedVerify follows the same full-row projection then per-group order.
            // Commit's _v* and all draft buffers, _proj/prepared packets, _x and _block
            // remain separate: none may be overwritten by these temporary phases.
            // Only this base goes into _owned. The ten views are never separately freed,
            // repointed, or resized, including when CUDA graphs or feature taps are used.
            SharedScratchBase = Alloc(layout.PoolBytes);
            SharedScratchPoolBytes = layout.PoolBytes;
            SharedScratchBytesSaved = layout.BytesSaved;
            foreach (var span in layout.Spans)
            {
                IntPtr pointer = SharedScratchPointer(SharedScratchBase, layout.PoolBytes, span);
                switch (span.Name)
                {
                    case "_xn": _xn = pointer; break;
                    case "_q": _q = pointer; break;
                    case "_gate": _gate = pointer; break;
                    case "_attn": _attn = pointer; break;
                    case "_act": _act = pointer; break;
                    case "_gdnQkv": _gdnQkv = pointer; break;
                    case "_gdnG": _gdnG = pointer; break;
                    case "_gdnBeta": _gdnBeta = pointer; break;
                    case "_gdnO": _gdnO = pointer; break;
                    case "_gdnY": _gdnY = pointer; break;
                    default: throw new InvalidOperationException($"[q35-cuda] unknown scratch view {span.Name}");
                }
            }
            Console.Error.WriteLine($"[q35-cuda] shared scratch enabled, pool={SharedScratchPoolBytes} bytes, saved={SharedScratchBytesSaved} bytes");
        }

        private void PublishScratchSpans(int rows)
        {
            ScratchBufferSpan Span(string name, string phase, IntPtr pointer, int width)
            {
                long bytes = checked((long)rows * width * sizeof(float));
                return new ScratchBufferSpan(name, phase, pointer, bytes,
                    SharedScratchEnabled ? AlignedScratchBytes(bytes) : Math.Max(ScratchAlignment, bytes));
            }
            ScratchBufferSpans = Array.AsReadOnly(new[] {
                Span("_xn", "normalize", _xn, _m.NEmbd),
                Span("_q", "attention", _q, QDim), Span("_gate", "attention", _gate, QDim),
                Span("_attn", "attention", _attn, QDim), Span("_act", "attention", _act, _m.NFf),
                Span("_gdnQkv", "gdn", _gdnQkv, ConvDim), Span("_gdnG", "gdn", _gdnG, _m.GdnVHeads),
                Span("_gdnBeta", "gdn", _gdnBeta, _m.GdnVHeads), Span("_gdnO", "gdn", _gdnO, ValueDim),
                Span("_gdnY", "gdn", _gdnY, ValueDim)
            });
        }
    }
}
