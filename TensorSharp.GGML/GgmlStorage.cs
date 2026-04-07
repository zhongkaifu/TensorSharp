// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using TensorSharp;

namespace TensorSharp.GGML
{
    [Serializable]
    public class GgmlStorage : Storage
    {
        private IntPtr buffer;

        public GgmlStorage(GgmlAllocator allocator, GgmlContext context, DType elementType, long elementCount)
            : base(allocator, elementType, elementCount)
        {
            Context = context ?? throw new ArgumentNullException(nameof(context));
            buffer = context.MemoryPool.Allocate(ByteLength);
        }

        public GgmlContext Context { get; }

        public int DeviceId => ((GgmlAllocator)Allocator).DeviceId;

        protected override void Destroy()
        {
            if (buffer != IntPtr.Zero)
            {
                Context.MemoryPool.Free(buffer, ByteLength);
                buffer = IntPtr.Zero;
            }
        }

        public override string LocationDescription()
        {
            return $"GGML:{DeviceId}";
        }

        public IntPtr PtrAtElement(long index)
        {
            return new IntPtr(buffer.ToInt64() + (index * ElementType.Size()));
        }

        public override int[] GetElementsAsInt(long index, int length)
        {
            unsafe
            {
                if (ElementType == DType.Int32)
                {
                    int* p = (int*)buffer.ToPointer();
                    int[] array = new int[length];

                    for (int i = 0; i < length; i++)
                    {
                        array[i] = *(p + index + i);
                    }

                    return array;
                }

                throw new NotSupportedException("Element type " + ElementType + " not supported");
            }
        }

        public override float GetElementAsFloat(long index)
        {
            unsafe
            {
                if (ElementType == DType.Float32)
                {
                    return ((float*)buffer.ToPointer())[index];
                }
                else if (ElementType == DType.Float64)
                {
                    return (float)((double*)buffer.ToPointer())[index];
                }
                else if (ElementType == DType.Int32)
                {
                    return ((int*)buffer.ToPointer())[index];
                }
                else if (ElementType == DType.UInt8)
                {
                    return ((byte*)buffer.ToPointer())[index];
                }

                throw new NotSupportedException("Element type " + ElementType + " not supported");
            }
        }

        public override float[] GetElementsAsFloat(long index, int length)
        {
            unsafe
            {
                if (ElementType == DType.Float32)
                {
                    float* p = (float*)buffer.ToPointer();
                    float[] array = new float[length];

                    for (int i = 0; i < length; i++)
                    {
                        array[i] = *(p + index + i);
                    }

                    return array;
                }

                throw new NotSupportedException("Element type " + ElementType + " not supported");
            }
        }

        public override void SetElementAsFloat(long index, float value)
        {
            unsafe
            {
                if (ElementType == DType.Float32)
                {
                    ((float*)buffer.ToPointer())[index] = value;
                }
                else if (ElementType == DType.Float64)
                {
                    ((double*)buffer.ToPointer())[index] = value;
                }
                else if (ElementType == DType.Int32)
                {
                    ((int*)buffer.ToPointer())[index] = (int)value;
                }
                else if (ElementType == DType.UInt8)
                {
                    ((byte*)buffer.ToPointer())[index] = (byte)value;
                }
                else
                {
                    throw new NotSupportedException("Element type " + ElementType + " not supported");
                }
            }
        }

        public override void SetElementsAsInt(long index, int[] value)
        {
            unsafe
            {
                if (ElementType == DType.Int32)
                {
                    for (int i = 0; i < value.Length; i++)
                    {
                        ((int*)buffer.ToPointer())[index + i] = value[i];
                    }
                }
                else
                {
                    throw new NotSupportedException("Element type " + ElementType + " not supported");
                }
            }
        }

        public override void SetElementsAsFloat(long index, float[] value)
        {
            unsafe
            {
                if (ElementType == DType.Float32)
                {
                    for (int i = 0; i < value.Length; i++)
                    {
                        ((float*)buffer.ToPointer())[index + i] = value[i];
                    }
                }
                else
                {
                    throw new NotSupportedException("Element type " + ElementType + " not supported");
                }
            }
        }

        public override void SetElementsAsHalf(long index, half[] value)
        {
            throw new NotSupportedException("GGML backends currently support Float32 tensors only. Disable AMP to use this backend.");
        }

        public override void CopyToStorage(long storageIndex, IntPtr src, long byteCount)
        {
            IntPtr dstPtr = PtrAtElement(storageIndex);
            unsafe
            {
                Buffer.MemoryCopy(src.ToPointer(), dstPtr.ToPointer(), byteCount, byteCount);
            }
        }

        public override void CopyFromStorage(IntPtr dst, long storageIndex, long byteCount)
        {
            IntPtr srcPtr = PtrAtElement(storageIndex);
            unsafe
            {
                Buffer.MemoryCopy(srcPtr.ToPointer(), dst.ToPointer(), byteCount, byteCount);
            }
        }
    }
}
