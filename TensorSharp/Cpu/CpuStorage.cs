using System;
using System.Runtime.InteropServices;

namespace TensorSharp.Cpu
{
    public class CpuStorage : Storage
    {
        public IntPtr buffer;


        public CpuStorage(IAllocator allocator, DType ElementType, long elementCount)
            : base(allocator, ElementType, elementCount)
        {
            buffer = Marshal.AllocHGlobal(new IntPtr(ByteLength));
        }

        protected override void Destroy()
        {
            Marshal.FreeHGlobal(buffer);
            buffer = IntPtr.Zero;
        }

        public override string LocationDescription()
        {
            return "CPU";
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
                    int* p = ((int*)buffer.ToPointer());
                    int[] array = new int[length];

                    for (int i = 0; i < length; i++)
                    {
                        array[i] = *(p + i);
                    }
                    return array;
                }
                else
                {
                    throw new NotSupportedException("Element type " + ElementType + " not supported");
                }
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
                else
                {
                    throw new NotSupportedException("Element type " + ElementType + " not supported");
                }
            }
        }

        public override float[] GetElementsAsFloat(long index, int length)
        {
            unsafe
            {
                if (ElementType == DType.Float32)
                {
                    float* p = ((float*)buffer.ToPointer());
                    float[] array = new float[length];

                    for (int i = 0; i < length; i++)
                    {
                        array[i] = *(p + i);
                    }
                    return array;
                }
                else
                {
                    throw new NotSupportedException("Element type " + ElementType + " not supported");
                }
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
            throw new NotImplementedException($"SetElementsAsHalf has not been implemented for CPUs yet.");
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
