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

        // Block-quantized types round their byte length up to the block boundary
        // (32-element blocks: Q8_0 = 34 bytes, Q4_0 = 18 bytes). Linear types stay
        // at ElementCount * ElementType.Size() as before.
        public long ByteLength => ElementType.ByteLengthFor(ElementCount);

        public bool IsOwnerExclusive()
        {
            return GetCurrentRefCount() == 1;
        }

        /// <summary>
        /// Ensure the device copy of this storage reflects any pending host-side
        /// writes (e.g. after <see cref="SetElementsAsFloat(long, float[])"/>).
        /// No-op for host-only storages; GPU storages upload the host buffer so
        /// subsequent kernels — including fused kernels that read the raw device
        /// pointer without a lazy sync — observe the written values. Used by the
        /// distributed AllReduce, which writes reduced results back through the
        /// host buffer.
        /// </summary>
        public virtual void EnsureDeviceCurrent() { }

        /// <summary>
        /// Supplies allocation metadata to backends whose device dimensions have
        /// a smaller range than the storage's total element count.
        /// </summary>
        protected internal virtual void SetTensorMetadata(ReadOnlySpan<long> sizes, ReadOnlySpan<long> strides) { }

        public abstract IntPtr PtrAtElement(long index);

        /// <summary>
        /// Host pointer for a caller that writes EVERY element of this storage and
        /// reads none of its previous contents. A device-backed storage may then skip
        /// refreshing its host mirror from the device; the written host data is
        /// uploaded before the next device use, as after <see cref="PtrAtElement"/>.
        /// </summary>
        public virtual IntPtr PtrAtElementForOverwrite(long index) => PtrAtElement(index);

        /// <summary>
        /// Host pointer for a caller that only READS through it (host decode attention over a
        /// K/V cache, a snapshot extract). A device-backed storage brings its host mirror up to
        /// date but does not treat the checkout as a host write, so the next device use does not
        /// upload the unchanged bytes again (and a device write that keeps the other elements,
        /// such as a K/V append, does not have to). Writing through the pointer is a bug: the
        /// write may never reach the device. Default: <see cref="PtrAtElement"/>.
        /// </summary>
        public virtual IntPtr PtrAtElementForRead(long index) => PtrAtElement(index);

        /// <summary>
        /// True, once, when this storage's memory was all zero at allocation and nothing has
        /// claimed that since: an allocation-time zero fill can then be skipped (a fresh
        /// anonymous mapping is zero already, and touching it page-faults every page). Call it
        /// right after allocation; later writes do not reset it. Default false.
        /// </summary>
        public virtual bool TakeAllocationZeroGuarantee() => false;


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

        /// <summary>
        /// Hook that backends with deferred GPU compute (currently the GGML/Metal
        /// backend in async mode) override to drain any in-flight work before host
        /// code reads or writes this storage's bytes directly. Default: no-op.
        ///
        /// Called from <see cref="TensorComputePrimitives.GetFloatPointer"/> /
        /// <see cref="TensorComputePrimitives.GetHalfPointer"/>, which are the
        /// gateways for raw-pointer host access on tensor data. Native op code
        /// goes through <see cref="PtrAtElement"/> directly and intentionally
        /// skips this hook so that op chaining stays asynchronous.
        /// </summary>
        public virtual void EnsureHostReadable()
        {
        }

    }
}
