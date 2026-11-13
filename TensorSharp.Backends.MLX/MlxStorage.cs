using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace TensorSharp.MLX
{
    [Serializable]
    public sealed unsafe class MlxStorage : Storage
    {
        private readonly object sync = new();
        private IntPtr buffer;
        private MlxNative.MlxArray deviceArray;
        // The device contents as consecutive flat arrays (an op's packed result whose pieces are read
        // separately): a view inside one part reads that part, anything else joins them first.
        private (MlxNative.MlxArray Array, long Offset, long Length)[] parts;
        private int[] deviceShape;
        private bool hostDirty = true;
        private bool deviceDirty;

        public MlxStorage(MlxAllocator allocator, DType elementType, long elementCount)
            : base(allocator, elementType, elementCount)
        {
            AllocatorImpl = allocator ?? throw new ArgumentNullException(nameof(allocator));
            if (ByteLength < 0)
                throw new ArgumentOutOfRangeException(nameof(elementCount));
        }

        internal MlxAllocator AllocatorImpl { get; }

        public int DeviceId => AllocatorImpl.DeviceId;

        protected override void SetTensorMetadata(ReadOnlySpan<long> sizes, ReadOnlySpan<long> strides)
        {
            if (ElementCount <= int.MaxValue || deviceShape != null)
                return;
            long count = 1;
            foreach (long size in sizes)
            {
                if (size < 0 || size > int.MaxValue) return;
                count = checked(count * size);
            }
            if (count == ElementCount)
                deviceShape = ToIntArray(sizes);
        }

        protected override void Destroy()
        {
            if (buffer != IntPtr.Zero)
            {
                NativeMemory.AlignedFree(buffer.ToPointer());
                buffer = IntPtr.Zero;
            }

            if (deviceArray.IsValid)
            {
                MlxNative.FreeArray(deviceArray);
                deviceArray = default;
            }
            FreeParts();
        }

        private void FreeParts()
        {
            if (parts == null) return;
            foreach (var part in parts) MlxNative.FreeArray(part.Array);
            parts = null;
        }

        /// <summary>The parts as one array (the storage's device array from now on). Caller holds sync.</summary>
        private void JoinParts()
        {
            if (parts == null) return;
            MlxNative.MlxArray joined = MlxNative.Retain(parts[0].Array);
            for (int i = 1; i < parts.Length; i++)
            {
                MlxNative.MlxArray next = MlxNative.ConcatenateAxis(joined, parts[i].Array, 0);
                MlxNative.FreeArray(joined);
                joined = next;
            }
            FreeParts();
            if (deviceArray.IsValid) MlxNative.FreeArray(deviceArray);
            deviceArray = joined;
            deviceShape = new[] { (int)ElementCount };
        }

        /// <summary>The part holding elements [first, last] of a view, or -1.</summary>
        private int PartOf(int[] shape, long[] strides, long offset)
        {
            long lo = offset, hi = offset;
            for (int i = 0; i < shape.Length; i++)
            {
                if (shape[i] == 0) return -1;
                long span = (shape[i] - 1L) * strides[i];
                if (span < 0) lo += span; else hi += span;
            }
            for (int p = 0; p < parts.Length; p++)
                if (lo >= parts[p].Offset && hi < parts[p].Offset + parts[p].Length) return p;
            return -1;
        }

        /// <summary>A view of one part, or (the view crosses parts) of the joined array.</summary>
        private bool TryPartView(int[] shape, long[] strides, long offset, out MlxNative.MlxArray view)
        {
            view = default;
            lock (sync)
            {
                if (parts == null) return false;
                int p = PartOf(shape, strides, offset);
                if (p < 0)
                {
                    JoinParts();
                    return false;
                }
                view = MlxNative.AsStrided(parts[p].Array, shape, strides, offset - parts[p].Offset);
                return true;
            }
        }

        /// <summary>Flat views (no copy) of elements [offset, offset + count): slices of the parts or of the
        /// device array. The caller frees them.</summary>
        internal MlxNative.MlxArray[] FlatRange(long offset, long count)
        {
            if (ElementCount > int.MaxValue) throw new NotSupportedException("MLX parts need storage under 2^31 elements.");
            ValidateElementRange(offset, count);
            lock (sync)
            {
                if (parts == null)
                {
                    EnsureDeviceCurrent(null);
                    MlxNative.MlxArray flat = MlxNative.Reshape(deviceArray, new[] { (int)ElementCount });
                    try { return new[] { MlxNative.Slice(flat, new[] { (int)offset }, new[] { (int)(offset + count) }, new[] { 1 }) }; }
                    finally { MlxNative.FreeArray(flat); }
                }
                var list = new List<MlxNative.MlxArray>();
                long end = offset + count;
                foreach (var part in parts)
                {
                    long from = Math.Max(offset, part.Offset), to = Math.Min(end, part.Offset + part.Length);
                    if (from >= to) continue;
                    list.Add(MlxNative.Slice(part.Array, new[] { (int)(from - part.Offset) }, new[] { (int)(to - part.Offset) }, new[] { 1 }));
                }
                return list.ToArray();
            }
        }

        /// <summary>
        /// Elements [offset, offset + n) become <paramref name="flat"/> (1-D arrays of n elements in all, taken
        /// over) with no copy; the rest keeps its contents as views of what held them.
        /// </summary>
        internal void AdoptRange(long offset, MlxNative.MlxArray[] flat)
        {
            long n = 0;
            foreach (var a in flat) n += MlxNative.GetArraySize(a);
            MlxNative.MlxArray[] head = offset > 0 ? FlatRange(0, offset) : Array.Empty<MlxNative.MlxArray>();
            MlxNative.MlxArray[] tail = offset + n < ElementCount ? FlatRange(offset + n, ElementCount - offset - n) : Array.Empty<MlxNative.MlxArray>();
            var all = new MlxNative.MlxArray[head.Length + flat.Length + tail.Length];
            head.CopyTo(all, 0);
            flat.CopyTo(all, head.Length);
            tail.CopyTo(all, head.Length + flat.Length);
            ReplaceDeviceParts(all);
        }

        /// <summary>
        /// Hold <paramref name="flatParts"/> (1-D arrays whose sizes add up to the storage) as the device
        /// contents without concatenating them; the storage takes ownership of the arrays.
        /// </summary>
        internal void ReplaceDeviceParts(params MlxNative.MlxArray[] flatParts)
        {
            if (ElementCount > int.MaxValue) throw new NotSupportedException("MLX parts need storage under 2^31 elements.");
            var list = new (MlxNative.MlxArray, long, long)[flatParts.Length];
            long at = 0;
            for (int i = 0; i < flatParts.Length; i++)
            {
                long n = MlxNative.GetArraySize(flatParts[i]);
                list[i] = (flatParts[i], at, n);
                at += n;
            }
            if (at != ElementCount)
            {
                foreach (var a in flatParts) MlxNative.FreeArray(a);
                throw new ArgumentException($"MLX parts hold {at} elements; the storage has {ElementCount}.", nameof(flatParts));
            }
            lock (sync)
            {
                if (deviceArray.IsValid) MlxNative.FreeArray(deviceArray);
                deviceArray = default;
                FreeParts();
                parts = list;
                deviceShape = new[] { (int)ElementCount };
                hostDirty = false;
                deviceDirty = true;
            }
        }

        public override string LocationDescription()
        {
            return $"MLX:{DeviceId}";
        }

        public override IntPtr PtrAtElement(long index)
        {
            ThrowIfDestroyed();
            ValidateElementRange(index, 0);
            EnsureHostReadable();
            hostDirty = true;
            return AddBytes(buffer, checked(index * ElementType.Size()));
        }

        public override void EnsureHostReadable()
        {
            ThrowIfDestroyed();
            lock (sync)
            {
                EnsureHostBufferAllocated();
                JoinParts();
                if (!deviceDirty || !deviceArray.IsValid || ByteLength == 0)
                    return;

                MlxNative.CopyArrayToHost(deviceArray, ElementType, buffer, ByteLength);
                deviceDirty = false;
                hostDirty = false;
            }
        }

        internal MlxNative.MlxArray CreateArrayView(Tensor tensor)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));
            if (!ReferenceEquals(tensor.Storage, this))
                throw new ArgumentException("Tensor is not backed by this MLX storage.", nameof(tensor));

            int[] shape = ToIntArray(tensor.Sizes);
            if (TryPartView(shape, ToLongArray(tensor.Strides), tensor.StorageOffset, out MlxNative.MlxArray partView))
                return partView;
            int[] storageShape = tensor.ElementCount() == ElementCount ? shape : null;
            if (storageShape == null && MlxBasicOps.TryGetStridedBox(tensor, out int[] parent, out _, out _))
                storageShape = parent;
            EnsureDeviceCurrent(storageShape);
            return MlxNative.AsStrided(deviceArray, shape, ToLongArray(tensor.Strides), tensor.StorageOffset, ElementCount > int.MaxValue);
        }

        /// <summary>A view of this storage in a shape no TensorSharp tensor describes (an
        /// extra leading batch axis, say), so it reaches MLX as a view rather than a copy.</summary>
        internal MlxNative.MlxArray CreateArrayView(int[] shape, long[] strides, long offset)
        {
            if (TryPartView(shape, strides, offset, out MlxNative.MlxArray partView))
                return partView;
            EnsureDeviceCurrent(shape);
            return MlxNative.AsStrided(deviceArray, shape, strides, offset, ElementCount > int.MaxValue);
        }

        internal void ReplaceDeviceArray(MlxNative.MlxArray array)
        {
            if (!array.IsValid)
                throw new ArgumentException("MLX array is empty.", nameof(array));
            // MLX dimensions are int, but its total array size and strides are
            // size_t. Flattening a large video activation would overflow one
            // dimension even though all of its original dimensions are valid.
            // Keep small storage flat for the common 1D slice-update path and
            // retain the original contiguous shape for larger storage.
            if (ElementCount > int.MaxValue && MlxNative.GetArraySize(array) != ElementCount)
                throw new ArgumentException("MLX array size does not match this storage.", nameof(array));
            MlxNative.MlxArray contiguous = ElementCount <= int.MaxValue
                ? MlxNative.Reshape(array, new[] { (int)ElementCount })
                : MlxNative.Contiguous(array);
            int[] shape;
            try
            {
                shape = ElementCount <= int.MaxValue
                    ? new[] { (int)ElementCount }
                    : MlxNative.GetArrayShape(contiguous);
            }
            catch
            {
                MlxNative.FreeArray(contiguous);
                throw;
            }
            MlxNative.FreeArray(array);

            lock (sync)
            {
                if (deviceArray.IsValid)
                    MlxNative.FreeArray(deviceArray);
                FreeParts();

                deviceArray = contiguous;
                deviceShape = shape;
                hostDirty = false;
                deviceDirty = true;
            }
        }

        internal void UpdateDeviceSlice(Tensor tensor, MlxNative.MlxArray update)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));
            if (!ReferenceEquals(tensor.Storage, this))
                throw new ArgumentException("Tensor is not backed by this MLX storage.", nameof(tensor));
            if (!update.IsValid)
                throw new ArgumentException("MLX update array is empty.", nameof(update));
            if (!tensor.IsContiguous())
                throw new NotSupportedException("MLX slice updates require contiguous tensor views.");
            ValidateElementRange(tensor.StorageOffset, tensor.ElementCount());

            if (ElementCount > int.MaxValue)
            {
                int[] shape = null;
                if (MlxBasicOps.TryGetStridedBox(tensor, out int[] parent, out _, out _))
                    shape = parent;
                EnsureDeviceCurrent(shape);
                UpdateLargeDeviceSlice(tensor.StorageOffset, tensor.ElementCount(), update);
                return;
            }

            EnsureDeviceCurrent();
            MlxNative.MlxArray flatUpdate = default;
            MlxNative.MlxArray updatedStorage = default;
            try
            {
                int length = (int)tensor.ElementCount();
                int start = (int)tensor.StorageOffset;
                flatUpdate = MlxNative.Reshape(update, new[] { length });
                updatedStorage = MlxNative.SliceUpdate(deviceArray, flatUpdate, start, checked(start + length));

                lock (sync)
                {
                    if (deviceArray.IsValid)
                        MlxNative.FreeArray(deviceArray);

                    deviceArray = updatedStorage;
                    updatedStorage = default;
                    hostDirty = false;
                    deviceDirty = true;
                }
            }
            finally
            {
                MlxNative.FreeArray(flatUpdate);
                MlxNative.FreeArray(updatedStorage);
            }
        }

        /// <summary>
        /// Write <paramref name="update"/> into a rectangular box of this storage viewed
        /// as a row-major array of <paramref name="parentShape"/> (which must cover the
        /// whole storage): rows [starts[i], starts[i] + update.shape[i]) of every axis.
        /// One device slice_update, where a strided view (a Narrow along an inner axis,
        /// say) otherwise has no GPU write path.
        /// </summary>
        internal void UpdateDeviceBox(int[] parentShape, int[] starts, int[] stops, MlxNative.MlxArray update)
        {
            if (!update.IsValid)
                throw new ArgumentException("MLX update array is empty.", nameof(update));

            EnsureDeviceCurrent(parentShape);
            MlxNative.MlxArray shaped = default;
            MlxNative.MlxArray updated = default;
            MlxNative.MlxArray flat = default;
            try
            {
                int[] strides = new int[parentShape.Length];
                Array.Fill(strides, 1);
                shaped = MlxNative.Reshape(deviceArray, parentShape);
                updated = MlxNative.SliceUpdateMulti(shaped, update, starts, stops, strides);
                int[] shape = ElementCount <= int.MaxValue ? new[] { (int)ElementCount } : parentShape;
                flat = MlxNative.Reshape(updated, shape);

                lock (sync)
                {
                    if (deviceArray.IsValid)
                        MlxNative.FreeArray(deviceArray);

                    deviceArray = flat;
                    deviceShape = (int[])shape.Clone();
                    flat = default;
                    hostDirty = false;
                    deviceDirty = true;
                }
            }
            finally
            {
                MlxNative.FreeArray(shaped);
                MlxNative.FreeArray(updated);
                MlxNative.FreeArray(flat);
            }
        }

        public override void EnsureDeviceCurrent()
        {
            EnsureDeviceCurrent(null);
        }

        private void EnsureDeviceCurrent(int[] shapeHint)
        {
            ThrowIfDestroyed();

            lock (sync)
            {
                JoinParts();
                if (deviceArray.IsValid && !hostDirty)
                    return;

                int[] shape = deviceShape;
                if (shape == null)
                {
                    if (ElementCount <= int.MaxValue)
                        shape = new[] { (int)ElementCount };
                    else if (shapeHint != null && ShapeElementCount(shapeHint) == ElementCount)
                        shape = (int[])shapeHint.Clone();
                    else
                        throw new NotSupportedException("Large MLX storage requires a tensor view whose dimensions describe the full storage.");
                }

                if (deviceArray.IsValid)
                {
                    MlxNative.FreeArray(deviceArray);
                    deviceArray = default;
                }

                if (buffer != IntPtr.Zero)
                    deviceArray = MlxNative.NewArrayFromHost(buffer, shape, ElementType);
                else
                    deviceArray = MlxNative.Full(shape, 0f, ElementType);
                deviceShape = shape;
                hostDirty = false;
                deviceDirty = false;
            }
        }

        private void UpdateLargeDeviceSlice(long start, long length, MlxNative.MlxArray update)
        {
            MlxNative.MlxArray updatedStorage = default;
            MlxNative.MlxArray contiguousUpdate = default;
            try
            {
                if (MlxNative.GetArraySize(update) != length)
                    throw new ArgumentException("MLX update size does not match the tensor view.", nameof(update));
                contiguousUpdate = MlxNative.Contiguous(update);
                updatedStorage = MlxNative.Retain(deviceArray);
                foreach (var box in GetSliceBoxes(deviceShape, start, length))
                {
                    MlxNative.MlxArray part = default;
                    MlxNative.MlxArray next = default;
                    try
                    {
                        part = MlxNative.AsStrided(contiguousUpdate, box.Shape, box.Strides, box.SourceOffset, largeStorage: true);
                        int[] steps = new int[deviceShape.Length];
                        Array.Fill(steps, 1);
                        next = MlxNative.SliceUpdateMulti(updatedStorage, part, box.Starts, box.Stops, steps);
                        MlxNative.FreeArray(updatedStorage);
                        updatedStorage = next;
                        next = default;
                    }
                    finally
                    {
                        MlxNative.FreeArray(part);
                        MlxNative.FreeArray(next);
                    }
                }

                lock (sync)
                {
                    MlxNative.FreeArray(deviceArray);
                    deviceArray = updatedStorage;
                    updatedStorage = default;
                    hostDirty = false;
                    deviceDirty = true;
                }
            }
            finally
            {
                MlxNative.FreeArray(updatedStorage);
                MlxNative.FreeArray(contiguousUpdate);
            }
        }

        internal readonly record struct SliceBox(int[] Starts, int[] Stops, int[] Shape, long[] Strides, long SourceOffset);

        // Cover a flat interval with row-major boxes without ever narrowing a
        // linear offset or a total element count to int. Each native slice index
        // is only a coordinate within an individual (int-sized) dimension.
        internal static IEnumerable<SliceBox> GetSliceBoxes(int[] shape, long start, long length)
        {
            long total = ShapeElementCount(shape);
            if (start < 0 || length < 0 || start > total || length > total - start)
                throw new ArgumentOutOfRangeException(nameof(start));
            long[] strides = new long[shape.Length];
            long stride = 1;
            for (int axis = shape.Length - 1; axis >= 0; --axis)
            {
                strides[axis] = stride;
                stride = checked(stride * shape[axis]);
            }
            long consumed = 0;
            while (consumed < length)
            {
                long position = start + consumed;
                int[] starts = new int[shape.Length];
                for (int axis = 0; axis < shape.Length; ++axis)
                {
                    starts[axis] = checked((int)(position / strides[axis]));
                    position %= strides[axis];
                }
                int split = shape.Length - 1;
                for (int axis = 0; axis < shape.Length; ++axis)
                {
                    if ((start + consumed) % strides[axis] == 0 && length - consumed >= strides[axis])
                    {
                        split = axis;
                        break;
                    }
                }
                int run = checked((int)Math.Min(shape[split] - starts[split], (length - consumed) / strides[split]));
                int[] stops = new int[shape.Length];
                int[] boxShape = new int[shape.Length];
                long[] boxStrides = new long[shape.Length];
                long boxCount = 1;
                for (int axis = shape.Length - 1; axis >= 0; --axis)
                {
                    stops[axis] = axis < split ? starts[axis] + 1 : axis == split ? starts[axis] + run : shape[axis];
                    boxShape[axis] = stops[axis] - starts[axis];
                    boxStrides[axis] = boxCount;
                    boxCount = checked(boxCount * boxShape[axis]);
                }
                yield return new SliceBox(starts, stops, boxShape, boxStrides, consumed);
                consumed += boxCount;
            }
        }

        private static long ShapeElementCount(int[] shape)
        {
            long count = 1;
            foreach (int dimension in shape)
            {
                if (dimension < 0) throw new ArgumentOutOfRangeException(nameof(shape));
                count = checked(count * dimension);
            }
            return count;
        }

        public override int[] GetElementsAsInt(long index, int length)
        {
            ValidateElementRange(index, length);
            if (ElementType != DType.Int32)
                throw new NotSupportedException("Element type " + ElementType + " not supported");

            int[] result = new int[length];
            EnsureHostReadable();
            int* src = (int*)AddBytes(buffer, checked(index * ElementType.Size()));
            for (int i = 0; i < length; i++)
                result[i] = src[i];
            return result;
        }

        public override void SetElementsAsInt(long index, int[] value)
        {
            if (value == null)
                throw new ArgumentNullException(nameof(value));
            ValidateElementRange(index, value.Length);
            if (ElementType != DType.Int32)
                throw new NotSupportedException("Element type " + ElementType + " not supported");

            int* dst = (int*)PtrAtElement(index);
            for (int i = 0; i < value.Length; i++)
                dst[i] = value[i];
        }

        public override float GetElementAsFloat(long index)
        {
            ValidateElementRange(index, 1);
            EnsureHostReadable();
            return ElementType switch
            {
                DType.Float32 => ((float*)buffer)[index],
                DType.Float64 => (float)((double*)buffer)[index],
                DType.Float16 => (float)((half*)buffer)[index],
                DType.Int32 => ((int*)buffer)[index],
                DType.UInt8 => ((byte*)buffer)[index],
                _ => throw new NotSupportedException("Element type " + ElementType + " not supported"),
            };
        }

        /// <summary>
        /// Bulk host read. Written as ONE <see cref="EnsureHostReadable"/> plus a
        /// single typed copy loop rather than <c>length</c> calls to
        /// <see cref="GetElementAsFloat"/>: every one of those calls re-entered
        /// <c>ThrowIfDestroyed()</c> + <c>lock (sync)</c>, and the hot caller here is
        /// ModelBase.TensorToFloatArray on the LM head, i.e. 202,048 locked
        /// per-element reads for EVERY decoded token of Muse-Glimmer.
        ///
        /// Like the per-element path this is a pure read: it must NOT raise
        /// <c>hostDirty</c> (which is why it uses <c>buffer</c> directly instead of
        /// <see cref="PtrAtElement"/>, whose checkout implies a pending host write).
        /// </summary>
        public override float[] GetElementsAsFloat(long index, int length)
        {
            ValidateElementRange(index, length);
            float[] result = new float[length];
            if (length == 0)
                return result;

            EnsureHostReadable();
            IntPtr src = AddBytes(buffer, checked(index * ElementType.Size()));
            switch (ElementType)
            {
                case DType.Float32:
                    new ReadOnlySpan<float>((float*)src, length).CopyTo(result);
                    break;
                case DType.Float64:
                {
                    double* typed = (double*)src;
                    for (int i = 0; i < length; i++)
                        result[i] = (float)typed[i];
                    break;
                }
                case DType.Float16:
                {
                    half* typed = (half*)src;
                    for (int i = 0; i < length; i++)
                        result[i] = typed[i];
                    break;
                }
                case DType.Int32:
                {
                    int* typed = (int*)src;
                    for (int i = 0; i < length; i++)
                        result[i] = typed[i];
                    break;
                }
                case DType.UInt8:
                {
                    byte* typed = (byte*)src;
                    for (int i = 0; i < length; i++)
                        result[i] = typed[i];
                    break;
                }
                default:
                    throw new NotSupportedException("Element type " + ElementType + " not supported");
            }

            return result;
        }

        public override void SetElementAsFloat(long index, float value)
        {
            ValidateElementRange(index, 1);
            EnsureHostReadable();
            switch (ElementType)
            {
                case DType.Float32:
                    ((float*)buffer)[index] = value;
                    break;
                case DType.Float64:
                    ((double*)buffer)[index] = value;
                    break;
                case DType.Float16:
                    ((half*)buffer)[index] = value;
                    break;
                case DType.Int32:
                    ((int*)buffer)[index] = (int)value;
                    break;
                case DType.UInt8:
                    ((byte*)buffer)[index] = (byte)value;
                    break;
                default:
                    throw new NotSupportedException("Element type " + ElementType + " not supported");
            }
            hostDirty = true;
        }

        /// <summary>
        /// Bulk host write, the mirror image of
        /// <see cref="GetElementsAsFloat"/>: one <see cref="EnsureHostReadable"/>
        /// (needed because a partial write must not discard the rest of a
        /// device-resident buffer) and one typed copy loop. The per-element
        /// spelling cost one <c>ThrowIfDestroyed()</c> + <c>lock (sync)</c> per
        /// element, which the Muse-Glimmer mmproj upload path hits ~1.85e9 times
        /// when its ~1.9B parameters are dequantized to F32.
        ///
        /// <c>hostDirty</c> is raised exactly once at the end, matching what the
        /// per-element path did on every write, so the next
        /// <see cref="EnsureDeviceCurrent"/> re-uploads.
        /// </summary>
        public override void SetElementsAsFloat(long index, float[] value)
        {
            if (value == null)
                throw new ArgumentNullException(nameof(value));
            ValidateElementRange(index, value.Length);
            int length = value.Length;
            if (length == 0)
                return;

            EnsureHostReadable();
            IntPtr dst = AddBytes(buffer, checked(index * ElementType.Size()));
            switch (ElementType)
            {
                case DType.Float32:
                    value.AsSpan().CopyTo(new Span<float>((float*)dst, length));
                    break;
                case DType.Float64:
                {
                    double* typed = (double*)dst;
                    for (int i = 0; i < length; i++)
                        typed[i] = value[i];
                    break;
                }
                case DType.Float16:
                {
                    half* typed = (half*)dst;
                    for (int i = 0; i < length; i++)
                        typed[i] = value[i];
                    break;
                }
                case DType.Int32:
                {
                    int* typed = (int*)dst;
                    for (int i = 0; i < length; i++)
                        typed[i] = (int)value[i];
                    break;
                }
                case DType.UInt8:
                {
                    byte* typed = (byte*)dst;
                    for (int i = 0; i < length; i++)
                        typed[i] = (byte)value[i];
                    break;
                }
                default:
                    // Thrown before hostDirty is raised, as in the per-element path.
                    throw new NotSupportedException("Element type " + ElementType + " not supported");
            }

            hostDirty = true;
        }

        public override void SetElementsAsHalf(long index, half[] value)
        {
            if (value == null)
                throw new ArgumentNullException(nameof(value));
            ValidateElementRange(index, value.Length);
            if (ElementType != DType.Float16)
                throw new NotSupportedException("Element type " + ElementType + " not supported");

            half* dst = (half*)PtrAtElement(index);
            for (int i = 0; i < value.Length; i++)
                dst[i] = value[i];
            hostDirty = true;
        }

        public override void CopyToStorage(long storageIndex, IntPtr src, long byteCount)
        {
            if (src == IntPtr.Zero && byteCount > 0)
                throw new ArgumentNullException(nameof(src));
            ValidateByteRange(storageIndex, byteCount);
            EnsureHostReadable();
            Buffer.MemoryCopy(src.ToPointer(), AddBytes(buffer, checked(storageIndex * ElementType.Size())).ToPointer(), byteCount, byteCount);
            hostDirty = true;
        }

        public override void CopyFromStorage(IntPtr dst, long storageIndex, long byteCount)
        {
            if (dst == IntPtr.Zero && byteCount > 0)
                throw new ArgumentNullException(nameof(dst));
            ValidateByteRange(storageIndex, byteCount);
            EnsureHostReadable();
            Buffer.MemoryCopy(AddBytes(buffer, checked(storageIndex * ElementType.Size())).ToPointer(), dst.ToPointer(), byteCount, byteCount);
        }

        private static int[] ToIntArray(ReadOnlySpan<long> values)
        {
            int[] result = new int[values.Length];
            for (int i = 0; i < values.Length; i++)
            {
                if (values[i] > int.MaxValue)
                    throw new NotSupportedException("MLX tensor dimensions larger than Int32.MaxValue are not supported yet.");
                result[i] = (int)values[i];
            }

            return result;
        }

        private static long[] ToLongArray(ReadOnlySpan<long> values)
        {
            long[] result = new long[values.Length];
            values.CopyTo(result);
            return result;
        }

        private void EnsureHostBufferAllocated()
        {
            if (buffer != IntPtr.Zero)
                return;

            nuint allocationBytes = (nuint)Math.Max(ByteLength, 1);
            buffer = (IntPtr)NativeMemory.AlignedAlloc(allocationBytes, 64);
            NativeMemory.Clear(buffer.ToPointer(), allocationBytes);
        }

        private void ValidateElementRange(long index, long length)
        {
            if (index < 0 || length < 0 || index > ElementCount || length > ElementCount - index)
                throw new ArgumentOutOfRangeException(nameof(index));
        }

        private void ValidateByteRange(long storageIndex, long byteCount)
        {
            if (byteCount < 0)
                throw new ArgumentOutOfRangeException(nameof(byteCount));

            long byteOffset = checked(storageIndex * ElementType.Size());
            if (byteOffset < 0 || byteOffset > ByteLength || byteCount > ByteLength - byteOffset)
                throw new ArgumentOutOfRangeException(nameof(storageIndex));
        }

        private static IntPtr AddBytes(IntPtr ptr, long bytes)
        {
            return new IntPtr(ptr.ToInt64() + bytes);
        }
    }
}
