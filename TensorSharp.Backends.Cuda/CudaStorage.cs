using System;
using System.Runtime.InteropServices;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    [Serializable]
    public sealed unsafe class CudaStorage : Storage
    {
        private readonly object sync = new object();
        private IntPtr hostBuffer;
        private IntPtr deviceBuffer;
        private bool hostDirty;
        private bool deviceDirty;

        public CudaStorage(CudaAllocator allocator, DType elementType, long elementCount)
            : base(allocator, elementType, elementCount)
        {
            AllocatorImpl = allocator ?? throw new ArgumentNullException(nameof(allocator));
            if (ByteLength < 0)
                throw new ArgumentOutOfRangeException(nameof(elementCount));

            long allocationSize = Math.Max(ByteLength, 1);
            hostBuffer = (IntPtr)NativeMemory.AlignedAlloc((nuint)allocationSize, 64);
            if (hostBuffer == IntPtr.Zero)
                throw new OutOfMemoryException($"Failed to allocate {allocationSize} bytes of CUDA host mirror memory.");

            AllocatorImpl.Context.MakeCurrent();
            CudaDriverApi.cuMemAlloc(out deviceBuffer, new UIntPtr((ulong)allocationSize)).ThrowOnError();
            CudaDriverApi.cuMemsetD8(deviceBuffer, 0, new UIntPtr((ulong)allocationSize)).ThrowOnError();
            NativeMemory.Clear(hostBuffer.ToPointer(), (nuint)allocationSize);
        }

        internal CudaAllocator AllocatorImpl { get; }

        internal IntPtr DeviceBuffer => deviceBuffer;

        public int DeviceId => AllocatorImpl.DeviceId;

        protected override void Destroy()
        {
            lock (sync)
            {
                if (deviceBuffer != IntPtr.Zero)
                {
                    AllocatorImpl.Context.MakeCurrent();
                    CudaDriverApi.cuMemFree(deviceBuffer);
                    deviceBuffer = IntPtr.Zero;
                }

                if (hostBuffer != IntPtr.Zero)
                {
                    NativeMemory.AlignedFree(hostBuffer.ToPointer());
                    hostBuffer = IntPtr.Zero;
                }
            }
        }

        public override string LocationDescription()
        {
            return $"CUDA:{DeviceId}";
        }

        public override IntPtr PtrAtElement(long index)
        {
            ThrowIfDisposed();
            ValidateElementRange(index, 0);

            // Existing TensorSharp model code may mutate through raw pointers. Treat
            // every pointer checkout as a possible host-side write so the next CUDA
            // dispatch refreshes device memory before using this storage as input.
            SyncHostFromDevice();
            hostDirty = true;
            return HostPtrAtElementUnchecked(index);
        }

        internal IntPtr DevicePtrAtElement(long index)
        {
            ThrowIfDisposed();
            ValidateElementRange(index, 0);
            return AddBytes(deviceBuffer, checked(index * ElementType.Size()));
        }

        internal void EnsureDeviceCurrent()
        {
            ThrowIfDisposed();
            if (ByteLength == 0)
                return;

            lock (sync)
            {
                if (!hostDirty)
                    return;

                AllocatorImpl.Context.MakeCurrent();
                CudaDriverApi.cuMemcpyHtoD(deviceBuffer, hostBuffer, new UIntPtr((ulong)ByteLength)).ThrowOnError();
                hostDirty = false;
                deviceDirty = false;
            }
        }

        internal void MarkDeviceModified()
        {
            ThrowIfDisposed();
            deviceDirty = true;
            hostDirty = false;
        }

        internal void SyncHostFromDevice()
        {
            ThrowIfDisposed();
            if (ByteLength == 0)
                return;

            lock (sync)
            {
                if (!deviceDirty)
                    return;

                AllocatorImpl.Context.MakeCurrent();
                AllocatorImpl.Stream.Synchronize();
                CudaDriverApi.cuMemcpyDtoH(hostBuffer, deviceBuffer, new UIntPtr((ulong)ByteLength)).ThrowOnError();
                deviceDirty = false;
                hostDirty = false;
            }
        }

        internal void CopyDeviceFrom(CudaStorage src)
        {
            if (src == null)
                throw new ArgumentNullException(nameof(src));
            if (src.ByteLength != ByteLength)
                throw new ArgumentException("CUDA device copy requires equal byte lengths.", nameof(src));

            src.EnsureDeviceCurrent();
            src.SynchronizeDeviceWork();
            AllocatorImpl.Context.MakeCurrent();
            CudaDriverApi.cuMemcpyDtoD(deviceBuffer, src.deviceBuffer, new UIntPtr((ulong)ByteLength)).ThrowOnError();
            MarkDeviceModified();
        }

        internal void SynchronizeDeviceWork()
        {
            ThrowIfDisposed();
            AllocatorImpl.Context.MakeCurrent();
            AllocatorImpl.Stream.Synchronize();
        }

        public override int[] GetElementsAsInt(long index, int length)
        {
            SyncHostFromDevice();
            if (ElementType != DType.Int32)
                throw new NotSupportedException("Element type " + ElementType + " not supported");

            ValidateElementRange(index, length);
            int[] array = new int[length];
            int* source = (int*)HostPtrAtElementUnchecked(index).ToPointer();
            for (int i = 0; i < length; i++)
                array[i] = source[i];

            return array;
        }

        public override void SetElementsAsInt(long index, int[] value)
        {
            if (value == null)
                throw new ArgumentNullException(nameof(value));
            if (ElementType != DType.Int32)
                throw new NotSupportedException("Element type " + ElementType + " not supported");

            ValidateElementRange(index, value.Length);
            int* target = (int*)HostPtrAtElementUnchecked(index).ToPointer();
            for (int i = 0; i < value.Length; i++)
                target[i] = value[i];

            hostDirty = true;
            deviceDirty = false;
        }

        public override float GetElementAsFloat(long index)
        {
            SyncHostFromDevice();
            ValidateElementRange(index, 1);

            return ElementType switch
            {
                DType.Float32 => ((float*)hostBuffer.ToPointer())[index],
                DType.Float64 => (float)((double*)hostBuffer.ToPointer())[index],
                DType.Int32 => ((int*)hostBuffer.ToPointer())[index],
                DType.UInt8 => ((byte*)hostBuffer.ToPointer())[index],
                _ => throw new NotSupportedException("Element type " + ElementType + " not supported"),
            };
        }

        public override float[] GetElementsAsFloat(long index, int length)
        {
            SyncHostFromDevice();
            if (ElementType != DType.Float32)
                throw new NotSupportedException("Element type " + ElementType + " not supported");

            ValidateElementRange(index, length);
            float[] array = new float[length];
            float* source = (float*)HostPtrAtElementUnchecked(index).ToPointer();
            for (int i = 0; i < length; i++)
                array[i] = source[i];

            return array;
        }

        public override void SetElementAsFloat(long index, float value)
        {
            ValidateElementRange(index, 1);
            switch (ElementType)
            {
                case DType.Float32:
                    ((float*)hostBuffer.ToPointer())[index] = value;
                    break;
                case DType.Float64:
                    ((double*)hostBuffer.ToPointer())[index] = value;
                    break;
                case DType.Int32:
                    ((int*)hostBuffer.ToPointer())[index] = (int)value;
                    break;
                case DType.UInt8:
                    ((byte*)hostBuffer.ToPointer())[index] = (byte)value;
                    break;
                default:
                    throw new NotSupportedException("Element type " + ElementType + " not supported");
            }

            hostDirty = true;
            deviceDirty = false;
        }

        public override void SetElementsAsFloat(long index, float[] value)
        {
            if (value == null)
                throw new ArgumentNullException(nameof(value));
            if (ElementType != DType.Float32)
                throw new NotSupportedException("Element type " + ElementType + " not supported");

            ValidateElementRange(index, value.Length);
            float* target = (float*)HostPtrAtElementUnchecked(index).ToPointer();
            for (int i = 0; i < value.Length; i++)
                target[i] = value[i];

            hostDirty = true;
            deviceDirty = false;
        }

        public override void SetElementsAsHalf(long index, half[] value)
        {
            throw new NotSupportedException("CUDA storage currently supports TensorSharp Float32/Float64/Int32/UInt8 host access.");
        }

        public override void CopyToStorage(long storageIndex, IntPtr src, long byteCount)
        {
            if (src == IntPtr.Zero && byteCount > 0)
                throw new ArgumentNullException(nameof(src));

            ValidateByteRange(storageIndex, byteCount);
            Buffer.MemoryCopy(src.ToPointer(), HostPtrAtElementUnchecked(storageIndex).ToPointer(), byteCount, byteCount);
            hostDirty = true;
            deviceDirty = false;
        }

        public override void CopyFromStorage(IntPtr dst, long storageIndex, long byteCount)
        {
            if (dst == IntPtr.Zero && byteCount > 0)
                throw new ArgumentNullException(nameof(dst));

            SyncHostFromDevice();
            ValidateByteRange(storageIndex, byteCount);
            Buffer.MemoryCopy(HostPtrAtElementUnchecked(storageIndex).ToPointer(), dst.ToPointer(), byteCount, byteCount);
        }

        private IntPtr HostPtrAtElementUnchecked(long index)
        {
            return AddBytes(hostBuffer, checked(index * ElementType.Size()));
        }

        private static IntPtr AddBytes(IntPtr pointer, long byteOffset)
        {
            return new IntPtr(pointer.ToInt64() + byteOffset);
        }

        private void ValidateElementRange(long index, long length)
        {
            if (index < 0 || length < 0 || index + length > ElementCount)
                throw new ArgumentOutOfRangeException(nameof(index));
        }

        private void ValidateByteRange(long storageIndex, long byteCount)
        {
            long byteOffset = checked(storageIndex * ElementType.Size());
            if (storageIndex < 0 || byteCount < 0 || byteOffset + byteCount > ByteLength)
                throw new ArgumentOutOfRangeException(nameof(storageIndex));
        }

        private void ThrowIfDisposed()
        {
            if (hostBuffer == IntPtr.Zero || deviceBuffer == IntPtr.Zero)
                throw new ObjectDisposedException(nameof(CudaStorage));
        }
    }
}
