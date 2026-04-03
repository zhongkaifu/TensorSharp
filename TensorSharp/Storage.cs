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

namespace TensorSharp
{
    [Serializable]
    public abstract class Storage : RefCounted
    {
        public Storage(IAllocator allocator, DType elementType, long elementCount)
        {
            Allocator = allocator;
            ElementType = elementType;
            ElementCount = elementCount;
        }

        /// <summary>
        /// Gets a reference to the allocator that constructed this Storage object.
        /// </summary>
        public IAllocator Allocator { get; private set; }

        public DType ElementType { get; private set; }
        public long ElementCount { get; private set; }

        public long ByteLength => ElementCount * ElementType.Size();

        public bool IsOwnerExclusive()
        {
            return GetCurrentRefCount() == 1;
        }


        public abstract int[] GetElementsAsInt(long index, int length);
        public abstract void SetElementsAsInt(long index, int[] value);


        public abstract string LocationDescription();

        public abstract float GetElementAsFloat(long index);
        public abstract float[] GetElementsAsFloat(long index, int length);
        public abstract void SetElementAsFloat(long index, float value);
        public abstract void SetElementsAsFloat(long index, float[] value);

        public abstract void SetElementsAsHalf(long index, half[] value);

        public abstract void CopyToStorage(long storageIndex, IntPtr src, long byteCount);
        public abstract void CopyFromStorage(IntPtr dst, long storageIndex, long byteCount);

    }
}
