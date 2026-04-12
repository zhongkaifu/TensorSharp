// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
namespace TensorSharp.Models
{
    internal sealed class BackendExecutionPlan : IBackendExecutionPlan
    {
        public BackendExecutionPlan(BackendType backendType)
        {
            BackendType = backendType;
        }

        public BackendType BackendType { get; }

        public bool UsesGgmlBackend =>
            BackendType == BackendType.GgmlCpu ||
            BackendType == BackendType.GgmlMetal ||
            BackendType == BackendType.GgmlCuda;

        public bool ShouldStoreWeightQuantized(GgufTensorInfo info)
        {
            return ModelBase.ShouldStoreWeightQuantized(BackendType, info);
        }
    }
}

