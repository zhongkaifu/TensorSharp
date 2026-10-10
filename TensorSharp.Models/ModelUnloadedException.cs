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

namespace TensorSharp.Models
{
    /// <summary>
    /// The model a request was using was unloaded under it (a model switch or a reload),
    /// so the request stopped where it was instead of running on freed weights. Not an
    /// <see cref="OperationCanceledException"/>: a request nobody cancelled must be told
    /// what happened, where a cancellation reads to a server as the client going away.
    /// A host whose own switch stopped the request reports it as stopped.
    /// </summary>
    public sealed class ModelUnloadedException : InvalidOperationException
    {
        public ModelUnloadedException()
            : base("The model was unloaded while this request was using it. Send it again.")
        {
        }

        public ModelUnloadedException(Exception innerException)
            : base("The model was unloaded while this request was using it. Send it again.", innerException)
        {
        }
    }
}
