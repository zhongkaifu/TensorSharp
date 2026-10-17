// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The seam a model that applies LoRA plug-ins sits behind, so the chat service and the
// hosts change or check a model's plug-ins without type-testing each family.
using System.Collections.Generic;

namespace TensorSharp.Models
{
    /// <summary>A model that applies TensorSharp LoRA plug-ins (<c>--lora</c>, <c>TS_LORAS</c>).</summary>
    public interface ILoraPluginHost
    {
        /// <summary>The plug-in config type this model takes, one of
        /// <see cref="LoraCliFlags.PluginTypes"/>.</summary>
        string LoraPluginType { get; }

        /// <summary>The resolved plug-ins in use (empty when none).</summary>
        IReadOnlyList<LoraSpec> LoraSpecs { get; }

        /// <summary>True when <paramref name="spec"/> is a plug-in this model can apply, judged by
        /// <see cref="LoraCliFlags.DeclaredPluginType"/> alone: it reads the plug-in's config,
        /// never its weights, and never downloads.</summary>
        /// <exception cref="System.IO.FileNotFoundException">A named file is missing.</exception>
        bool AcceptsLora(LoraSpec spec);

        /// <summary>Replace the plug-ins applied to every later request; an empty list removes
        /// them. A failure leaves the previous set in place.</summary>
        void SetLoras(IReadOnlyList<LoraSpec> specs);
    }
}
