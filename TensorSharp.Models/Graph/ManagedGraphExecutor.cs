// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Runtime.CompilerServices;

using TensorSharp.Cpu;
using TensorSharp.Graph;

namespace TensorSharp.Models.Graph
{
    /// <summary>
    /// The managed CPU backend's executor plugin: programs run node by node on host tensors, quantized
    /// products through the managed GEMM kernels (the reference every other executor is checked against).
    /// </summary>
    public sealed class ManagedGraphExecutor : TensorGraphExecutor
    {
        [ModuleInitializer]
        internal static void Register()
            => GraphExecutors.Register(typeof(CpuAllocator), allocator => new ManagedGraphExecutor(allocator));

        public ManagedGraphExecutor(IAllocator allocator) : base(allocator) { }

        protected override unsafe Tensor QuantizedMatMul(IGraphWeight weight, Tensor x)
        {
            if (weight.Data == IntPtr.Zero)
                throw new InvalidOperationException("a quantized weight's host bytes were released");
            int rows = (int)x.Sizes[0];
            var result = new Tensor(Allocator, DType.Float32, rows, weight.Ne1);
            ManagedQuantizedOps.AddmmQuantizedToFloat32(weight.Format, weight.Data, weight.Ne0, weight.Ne1,
                TensorComputePrimitives.GetFloatPointer(x), (int)weight.Ne0, rows,
                TensorComputePrimitives.GetFloatPointer(result), (int)weight.Ne1);
            return result;
        }
    }
}
