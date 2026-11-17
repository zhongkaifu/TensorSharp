// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

using TensorSharp.Cuda;
using TensorSharp.Graph;
using TensorSharp.MLX;
using TensorSharp.Runtime;

namespace TensorSharp.Models.Graph
{
    /// <summary>
    /// One weight of a <see cref="NetworkWeights"/> set: a product weight kept in its GGUF block format (rows of
    /// <see cref="RowWidth"/> values: what a <see cref="GraphOp.MulMat"/> or <see cref="GraphOp.GetRows"/> reads), or a dense
    /// F32 tensor on the set's allocator (norms, biases, conv kernels, and product weights the backend has no kernel for).
    /// </summary>
    public sealed class NetworkWeight
    {
        internal NetworkWeight(string name, long[] ne, int sourceFormat, QuantizedWeight quantized, Tensor dense, bool ownsQuantized = true)
        {
            Name = name;
            Ne = ne;
            SourceFormat = sourceFormat;
            Quantized = quantized;
            Dense = dense;
            OwnsQuantized = ownsQuantized;
        }

        public string Name { get; }

        /// <summary>The extents the weight was loaded with (ggml order: innermost first).</summary>
        public long[] Ne { get; }

        /// <summary>The GGUF tensor type it was read from (GraphFormat / GgmlTensorType ids).</summary>
        public int SourceFormat { get; }

        /// <summary>Kept in a block format (a product weight); otherwise <see cref="Dense"/> holds it as F32.</summary>
        public bool IsQuantized => Quantized != null;

        /// <summary>The format the graph reads: the block format, or F32.</summary>
        public int Format => Quantized?.GgmlType ?? GraphFormat.F32;

        public long Elements
        {
            get
            {
                long n = 1;
                foreach (long d in Ne) n *= d;
                return n;
            }
        }

        /// <summary>Values a row (ne0).</summary>
        public long RowWidth => Ne[0];

        /// <summary>Rows: every extent past ne0 folded.</summary>
        public long Rows => Elements / Ne[0];

        /// <summary>The product weight (rows of <see cref="RowWidth"/>), or null.</summary>
        public QuantizedWeight Quantized { get; }

        /// <summary>The F32 tensor (sizes the reverse of <see cref="Ne"/>), or null.</summary>
        public Tensor Dense { get; }

        internal bool OwnsQuantized { get; }
    }

    /// <summary>
    /// The weights of a non-decoder network (a vision tower, an audio encoder, a speech decoder) on one backend: read from a
    /// GGUF (an mmproj) or added from host values, held in host memory (product weights in their block format, so every
    /// executor makes its own resident copy on first use, as the decoder's weights) or as F32 tensors on the allocator. Whether
    /// a product weight stays in its block format follows the decoder's rule for the backend
    /// (<see cref="ModelBase.ShouldStoreWeightQuantized(BackendType, GgufTensorInfo)"/>): where the backend has no kernel for
    /// the format it is dequantized to F32 at load. Disposing releases the backends' resident copies and the host memory.
    /// </summary>
    public sealed class NetworkWeights : IDisposable
    {
        private readonly Dictionary<string, NetworkWeight> _weights = new(StringComparer.Ordinal);
        private readonly object _sync = new();
        private bool _disposed;

        public NetworkWeights(IAllocator allocator, BackendType backend)
        {
            Allocator = allocator ?? throw new ArgumentNullException(nameof(allocator));
            Backend = backend;
        }

        public IAllocator Allocator { get; }

        public BackendType Backend { get; }

        public int Count
        {
            get { lock (_sync) return _weights.Count; }
        }

        /// <summary>Bytes held in block formats (host copies).</summary>
        public long QuantizedBytes { get; private set; }

        /// <summary>Bytes held as dense F32 tensors.</summary>
        public long DenseBytes { get; private set; }

        public IReadOnlyCollection<string> Names
        {
            get { lock (_sync) return new List<string>(_weights.Keys); }
        }

        public bool Contains(string name)
        {
            lock (_sync) return _weights.ContainsKey(name);
        }

        public bool TryGet(string name, out NetworkWeight weight)
        {
            lock (_sync) return _weights.TryGetValue(name, out weight);
        }

        public NetworkWeight Get(string name)
        {
            lock (_sync)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                if (_weights.TryGetValue(name, out NetworkWeight w)) return w;
            }
            throw new KeyNotFoundException($"network weight '{name}' is not loaded");
        }

        /// <summary>The weight called <paramref name="name"/>, made by <paramref name="make"/> on first use (a re-laid or
        /// derived weight: an im2col kernel, a pre-summed patch kernel).</summary>
        public NetworkWeight GetOrAdd(string name, Func<NetworkWeights, NetworkWeight> make)
        {
            lock (_sync)
                if (_weights.TryGetValue(name, out NetworkWeight w)) return w;
            NetworkWeight made = make(this);
            if (!ReferenceEquals(made, Get(name)))
                throw new InvalidOperationException($"the maker of network weight '{name}' added another name");
            return made;
        }

        /// <summary>Load every tensor of <paramref name="file"/> whose name <paramref name="include"/> accepts (null: all);
        /// returns how many were loaded.</summary>
        public int LoadGguf(GgufFile file, Func<string, bool> include = null)
        {
            ArgumentNullException.ThrowIfNull(file);
            int loaded = 0;
            foreach (GgufTensorInfo info in file.Tensors.Values)
            {
                if (include != null && !include(info.Name)) continue;
                LoadTensor(file, info.Name);
                loaded++;
            }
            return loaded;
        }

        /// <summary>Load tensor <paramref name="name"/> of <paramref name="file"/>.</summary>
        public NetworkWeight LoadTensor(GgufFile file, string name)
        {
            ArgumentNullException.ThrowIfNull(file);
            if (!file.Tensors.TryGetValue(name, out GgufTensorInfo info))
                throw new KeyNotFoundException($"the GGUF has no tensor '{name}'");
            long[] ne = new long[info.Shape.Length];
            for (int i = 0; i < ne.Length; i++) ne[i] = (long)info.Shape[i];
            long elements = info.NumElements;
            long rows = elements / Math.Max(1, ne[0]);
            var flat = new GgufTensorInfo { Name = info.Name, Type = info.Type, Offset = info.Offset, Shape = new[] { (ulong)ne[0], (ulong)rows } };
            if (ne.Length >= 2 && info.Type != GgmlTensorType.F32 && IsProductShape(info.Type, ne[0]) && KeepsBlocks(flat))
            {
                long bytes = file.GetTensorByteCount(info);
                IntPtr ptr = QuantizedWeight.AllocateBuffer(bytes);
                try
                {
                    file.ReadTensorDataToNative(info, ptr, bytes);
                }
                catch
                {
                    QuantizedWeight.FreeBuffer(ptr);
                    throw;
                }
                var q = new QuantizedWeight(ptr, bytes, (int)info.Type, ne[0], rows);
                return Add(new NetworkWeight(name, ne, (int)info.Type, q, null));
            }
            var values = new float[elements];
            if (info.Type == GgmlTensorType.F32)
            {
                file.ReadTensorDataToFloat32(info, values, elements);
            }
            else
            {
                byte[] raw = file.ReadTensorData(info);
                NativeDequant.DequantizeToFloat32((int)info.Type, raw, 0, values, 0, elements);
            }
            return Add(new NetworkWeight(name, ne, (int)info.Type, null, NewDense(values, ne)));
        }

        /// <summary>Add an F32 weight from host values with ggml extents <paramref name="ne"/> (innermost first).</summary>
        public NetworkWeight AddF32(string name, float[] values, params long[] ne)
        {
            ArgumentNullException.ThrowIfNull(values);
            if (ne == null || ne.Length == 0) ne = new long[] { values.LongLength };
            long n = 1;
            foreach (long d in ne) n *= d;
            if (n != values.LongLength) throw new ArgumentException($"network weight '{name}': {values.LongLength} values for extents [{string.Join(", ", ne)}]");
            return Add(new NetworkWeight(name, (long[])ne.Clone(), GraphFormat.F32, null, NewDense(values, ne)));
        }

        /// <summary>Add a product weight from the raw blocks of <paramref name="rows"/> rows of <paramref name="rowWidth"/>
        /// values in GGUF type <paramref name="ggmlType"/>. Where the backend has no kernel for the format it is stored
        /// dequantized, as <see cref="LoadTensor"/> stores it.</summary>
        public NetworkWeight AddRaw(string name, int ggmlType, ReadOnlySpan<byte> raw, long rowWidth, long rows)
        {
            long expected = RowBytes(ggmlType, rowWidth) * rows;
            if (raw.Length != expected) throw new ArgumentException($"network weight '{name}': {raw.Length} bytes for {rows} rows of {rowWidth} (expected {expected})");
            long[] ne = { rowWidth, rows };
            var flat = new GgufTensorInfo { Name = name, Type = (GgmlTensorType)ggmlType, Shape = new[] { (ulong)rowWidth, (ulong)rows } };
            if (ggmlType != GraphFormat.F32 && KeepsBlocks(flat))
            {
                IntPtr ptr = QuantizedWeight.AllocateBuffer(raw.Length);
                unsafe
                {
                    fixed (byte* src = raw) Buffer.MemoryCopy(src, (void*)ptr, raw.Length, raw.Length);
                }
                return Add(new NetworkWeight(name, ne, ggmlType, new QuantizedWeight(ptr, raw.Length, ggmlType, rowWidth, rows), null));
            }
            var values = new float[rowWidth * rows];
            NativeDequant.DequantizeToFloat32(ggmlType, raw.ToArray(), 0, values, 0, values.LongLength);
            return Add(new NetworkWeight(name, ne, ggmlType, null, NewDense(values, ne)));
        }

        /// <summary>
        /// Weight <paramref name="name"/> read as <paramref name="rowWidth"/>-wide rows (a 4-D patch kernel as a
        /// [KW * KH * C, out] matrix): the same bytes, a weight of its own (named "name@width"; executors that key resident
        /// copies on the weight make one for it). A block-format weight keeps whole blocks a row.
        /// </summary>
        public NetworkWeight Reshaped(string name, long rowWidth)
        {
            NetworkWeight w = Get(name);
            if (w.RowWidth == rowWidth) return w;
            if (w.Elements % rowWidth != 0) throw new ArgumentException($"network weight '{name}' ({w.Elements} values) is not rows of {rowWidth}");
            string key = $"{name}@{rowWidth}";
            return GetOrAdd(key, set =>
            {
                long[] ne = { rowWidth, w.Elements / rowWidth };
                if (w.Quantized == null)
                {
                    using Tensor view = w.Dense.View(ne[1], ne[0]);
                    return set.Add(new NetworkWeight(key, ne, w.SourceFormat, null, view.CopyRef()));
                }
                long block = GgufFile.GetBlockSize((GgmlTensorType)w.Format);
                if (rowWidth % block != 0 || w.RowWidth % block != 0)
                    throw new ArgumentException($"network weight '{name}' ({(GgmlTensorType)w.Format}) cannot be read as rows of {rowWidth}: rows hold whole blocks of {block}");
                var view2 = QuantizedWeight.CreateExternalView(w.Quantized.Data, w.Quantized.RawBytes, w.Format, ne[0], ne[1], w.Quantized);
                return set.Add(new NetworkWeight(key, ne, w.SourceFormat, view2, null, ownsQuantized: false));
            });
        }

        /// <summary>
        /// Rows [<paramref name="firstRow"/>, + <paramref name="rows"/>) of product weight <paramref name="name"/> as a weight
        /// of their own (named "name#first+rows"): a product with it computes those rows only (one codebook's logits of a
        /// stacked table) on every executor. A dense weight's block is a view of its tensor.
        /// </summary>
        public NetworkWeight RowBlock(string name, long firstRow, long rows)
        {
            NetworkWeight w = Get(name);
            if (firstRow < 0 || rows < 1 || firstRow + rows > w.Rows)
                throw new ArgumentOutOfRangeException(nameof(rows), $"rows [{firstRow}, {firstRow + rows}) of network weight '{name}' ({w.Rows} rows)");
            if (firstRow == 0 && rows == w.Rows) return w;
            string key = $"{name}#{firstRow}+{rows}";
            return GetOrAdd(key, set =>
            {
                long[] ne = { w.RowWidth, rows };
                if (w.Quantized == null)
                {
                    using Tensor all = w.Dense.View(w.Rows, w.RowWidth);
                    using Tensor block = all.Narrow(0, firstRow, rows);
                    return set.Add(new NetworkWeight(key, ne, w.SourceFormat, null, block.CopyRef()));
                }
                long rowBytes = w.Quantized.RawBytes / w.Rows;
                var view = QuantizedWeight.CreateExternalView(w.Quantized.Data + (nint)(firstRow * rowBytes), rowBytes * rows, w.Format,
                    w.RowWidth, rows, w.Quantized);
                return set.Add(new NetworkWeight(key, ne, w.SourceFormat, view, null, ownsQuantized: false));
            });
        }

        /// <summary>Weight <paramref name="name"/>'s values as F32 on the host (dequantized where it is held in a block
        /// format), in ggml order.</summary>
        public float[] ReadF32(string name)
        {
            NetworkWeight w = Get(name);
            var values = new float[w.Elements];
            if (w.Quantized != null)
                NativeDequant.DequantizeToFloat32(w.Format, w.Quantized.Data, values, 0, values.LongLength);
            else
                w.Dense.CopyToArray(values);
            return values;
        }

        private Tensor NewDense(float[] values, long[] ne)
        {
            var sizes = new long[ne.Length];
            for (int i = 0; i < ne.Length; i++) sizes[i] = ne[ne.Length - 1 - i];
            var t = new Tensor(Allocator, DType.Float32, sizes);
            t.SetElementsAsFloat(values);
            return t;
        }

        private NetworkWeight Add(NetworkWeight w)
        {
            lock (_sync)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                if (_weights.ContainsKey(w.Name))
                {
                    Release(w);
                    throw new InvalidOperationException($"network weight '{w.Name}' is already loaded");
                }
                _weights[w.Name] = w;
                if (w.Quantized != null && w.OwnsQuantized) QuantizedBytes += w.Quantized.RawBytes;
                if (w.Dense != null) DenseBytes += w.Elements * sizeof(float);
                return w;
            }
        }

        private static readonly HashSet<(BackendType, GgmlTensorType)> s_denseNoted = new();

        /// <summary>
        /// Whether a product weight stays in its block format here: the decoder's rule for the backend, and on MLX only formats
        /// its executor multiplies on the device (MlxGraphExecutor takes any other format's products on the host, a CPU GEMM a
        /// call: an F16 mmproj's projections there ran ~40x slower). A format that is not kept is held as F32, noted once.
        /// </summary>
        private bool KeepsBlocks(GgufTensorInfo flat)
        {
            bool keep = Backend == BackendType.Mlx
                ? MlxQuantizedOps.CanPreloadDirectFloatType((int)flat.Type)
                : ModelBase.ShouldStoreWeightQuantized(Backend, flat);
            if (!keep)
            {
                lock (s_denseNoted)
                    if (s_denseNoted.Add((Backend, flat.Type)))
                        Console.WriteLine($"  network weights: {flat.Type} products run on {Backend} as F32 copies (its graph executor has no " +
                            $"{flat.Type} product kernel on the device): those weights take {4.0 * GgufFile.GetBlockSize(flat.Type) / GgufFile.GetTypeSize(flat.Type):F1}x their file size.");
            }
            return keep;
        }

        /// <summary>Whether a tensor of rows <paramref name="width"/> wide is a product weight: whole 32-value blocks a row (every
        /// projection; a conv kernel's [3, 3, 1, C] or a patch kernel's [16, 16, 3, C] stays a dense tensor the graph re-reads in
        /// any shape).</summary>
        private static bool IsProductShape(GgmlTensorType type, long width)
            => width >= 32 && width % 32 == 0 && width % GgufFile.GetBlockSize(type) == 0;

        /// <summary>Row bytes of <paramref name="width"/> values of a GGUF type.</summary>
        internal static long RowBytes(int ggmlType, long width)
        {
            var type = (GgmlTensorType)ggmlType;
            long block = GgufFile.GetBlockSize(type);
            if (width % block != 0) throw new ArgumentException($"{type} rows hold whole blocks of {block} values; {width} is not");
            return width / block * GgufFile.GetTypeSize(type);
        }

        private void Release(NetworkWeight w)
        {
            if (w.Quantized != null)
            {
                QuantizedWeight q = w.Quantized;
                // Every backend's resident copy keyed on the weight goes first, then the host bytes it was made from.
                if (Allocator is CudaAllocator cuda) CudaQuantizedOps.ReleaseQuantizedWeight(cuda, q.CacheKey);
                else if (Allocator is MlxAllocator mlx) MlxQuantizedOps.ReleaseQuantizedWeight(mlx, q.CacheKey);
                else if (q.Data != IntPtr.Zero && GraphExecutors.For(Allocator) is { Capabilities.BindsHostPointers: true } executor)
                    executor.Invalidate(q.Data);
                q.Dispose();
            }
            if (w.Dense != null)
            {
                if (GraphExecutors.For(Allocator) is { Capabilities.BindsHostPointers: true } executor)
                    executor.Invalidate(TensorComputePrimitives.GetStoragePointer(w.Dense));
                w.Dense.Dispose();
            }
        }

        public void Dispose()
        {
            List<NetworkWeight> all;
            lock (_sync)
            {
                if (_disposed) return;
                _disposed = true;
                all = new List<NetworkWeight>(_weights.Values);
                _weights.Clear();
            }
            // Views first: they borrow their owners' bytes.
            foreach (NetworkWeight w in all) if (!w.OwnsQuantized || (w.Dense != null && w.Quantized == null)) Release(w);
            foreach (NetworkWeight w in all) if (w.OwnsQuantized && w.Quantized != null) Release(w);
        }
    }
}
