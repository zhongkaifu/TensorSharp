using System;
using System.Runtime.InteropServices;

namespace TensorSharp.MLX
{
    [Serializable]
    public sealed unsafe class MlxStorage : Storage
    {
        private readonly object sync = new();
        private IntPtr buffer;
        private MlxNative.MlxArray deviceArray;
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

            EnsureDeviceCurrent();
            return MlxNative.AsStrided(deviceArray, ToIntArray(tensor.Sizes), ToLongArray(tensor.Strides), tensor.StorageOffset);
        }

        internal void ReplaceDeviceArray(MlxNative.MlxArray array)
        {
            if (!array.IsValid)
                throw new ArgumentException("MLX array is empty.", nameof(array));
            if (ElementCount > int.MaxValue)
                throw new NotSupportedException("MLX storage arrays larger than Int32.MaxValue elements are not supported yet.");

            // Phase 4 attempted to skip the reshape when the incoming array
            // was already row-contiguous (and stored the multi-dim array
            // directly as the storage's deviceArray). That broke
            // <see cref="UpdateDeviceSlice"/>, which calls 1D
            // <c>SliceUpdate</c> on <c>deviceArray</c> — mlx errors out
            // with "Invalid number of indices or strides for array with
            // dimension 2" when the deviceArray is multi-dim. Keep the
            // invariant: <c>deviceArray</c> is always stored as 1D flat,
            // and reshape on entry. The reshape on a row-contiguous array
            // is internally a metadata-only no-op in MLX, so the cost is
            // limited to the mlx_reshape C call itself (~5 µs).
            MlxNative.MlxArray flat = MlxNative.Reshape(array, new[] { (int)ElementCount });
            MlxNative.FreeArray(array);

            lock (sync)
            {
                if (deviceArray.IsValid)
                    MlxNative.FreeArray(deviceArray);

                deviceArray = flat;
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
            if (tensor.StorageOffset < 0 || tensor.StorageOffset > int.MaxValue)
                throw new NotSupportedException("MLX slice offsets larger than Int32.MaxValue are not supported yet.");
            if (tensor.ElementCount() > int.MaxValue)
                throw new NotSupportedException("MLX slice lengths larger than Int32.MaxValue are not supported yet.");

            EnsureDeviceCurrent();
            MlxNative.MlxArray flatUpdate = default;
            MlxNative.MlxArray updatedStorage = default;
            try
            {
                int length = (int)tensor.ElementCount();
                int start = (int)tensor.StorageOffset;
                flatUpdate = MlxNative.Reshape(update, new[] { length });
                updatedStorage = MlxNative.SliceUpdate(deviceArray, flatUpdate, start, start + length);

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

        internal void EnsureDeviceCurrent()
        {
            ThrowIfDestroyed();
            if (ElementCount > int.MaxValue)
                throw new NotSupportedException("MLX storage arrays larger than Int32.MaxValue elements are not supported yet.");

            lock (sync)
            {
                if (deviceArray.IsValid && !hostDirty)
                    return;

                if (deviceArray.IsValid)
                {
                    MlxNative.FreeArray(deviceArray);
                    deviceArray = default;
                }

                if (buffer != IntPtr.Zero)
                    deviceArray = MlxNative.NewArrayFromHost(buffer, new[] { (int)ElementCount }, ElementType);
                else
                    deviceArray = MlxNative.Full(new[] { (int)ElementCount }, 0f, ElementType);
                hostDirty = false;
                deviceDirty = false;
            }
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

        public override float[] GetElementsAsFloat(long index, int length)
        {
            ValidateElementRange(index, length);
            float[] result = new float[length];
            for (int i = 0; i < length; i++)
                result[i] = GetElementAsFloat(index + i);
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

        public override void SetElementsAsFloat(long index, float[] value)
        {
            if (value == null)
                throw new ArgumentNullException(nameof(value));
            ValidateElementRange(index, value.Length);
            for (int i = 0; i < value.Length; i++)
                SetElementAsFloat(index + i, value[i]);
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
            if (index < 0 || length < 0 || index + length > ElementCount)
                throw new ArgumentOutOfRangeException(nameof(index));
        }

        private void ValidateByteRange(long storageIndex, long byteCount)
        {
            if (byteCount < 0)
                throw new ArgumentOutOfRangeException(nameof(byteCount));

            long byteOffset = checked(storageIndex * ElementType.Size());
            if (byteOffset < 0 || byteOffset + byteCount > ByteLength)
                throw new ArgumentOutOfRangeException(nameof(storageIndex));
        }

        private static IntPtr AddBytes(IntPtr ptr, long bytes)
        {
            return new IntPtr(ptr.ToInt64() + bytes);
        }
    }
}
