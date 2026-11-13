// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using TensorSharp.Models.Architecture;

namespace TensorSharp.Models
{
    /// <summary>Hunyuan dense (Hy-MT / Hunyuan-1.8B-style) architecture plug-in.</summary>
    internal static class HunyuanDenseArchitecture
    {
        public static ModelArchitectureDescriptor Descriptor { get; } = new()
        {
            Id = "hunyuan-dense",
            DisplayName = "Hunyuan Dense",
            Aliases = ["hunyuan-dense"],
            Factory = c => new HunyuanDenseModel(c.GgufPath, c.Backend, c.TpDegree, c.TpGroup, c.LayerSplitDegree),
            Serving = ServingContract.SharedRuntime(hasRoutedExperts: false),
        };
    }
}
