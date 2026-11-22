// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;

namespace TensorSharp.Models
{
    /// <summary>
    /// The process-wide request for <c>--nvfp4-prompt-activations fp4|q8</c>: how an NVFP4 weight's prompt-row products
    /// quantize their activations (<see cref="Nvfp4PromptActivations"/>). The CLI and server front-ends call
    /// <see cref="Set"/> once at start-up; each model captures <see cref="Current"/> at construction and applies it to its
    /// own products, so a long-lived process can load models made before and after a change. No environment variable.
    /// </summary>
    public static class Nvfp4PromptActivationsConfig
    {
        /// <summary>The command-line option.</summary>
        public const string OptionName = "--nvfp4-prompt-activations";

        private static volatile int _current = (int)Nvfp4PromptActivations.Fp4;
        // A test's own choice for the models it constructs on its flow, ahead of the process-wide one, which tests running
        // alongside may set (ForCurrentFlowInTests).
        private static readonly System.Threading.AsyncLocal<Nvfp4PromptActivations?> s_flow = new();

        /// <summary>The choice models constructed now capture: <see cref="Nvfp4PromptActivations.Fp4"/> unless set.</summary>
        public static Nvfp4PromptActivations Current => s_flow.Value ?? (Nvfp4PromptActivations)_current;

        public static void Set(Nvfp4PromptActivations value) => _current = (int)value;

        /// <summary><c>fp4</c> or <c>q8</c> (any case).</summary>
        public static bool TryParse(string value, out Nvfp4PromptActivations activations)
        {
            activations = Nvfp4PromptActivations.Fp4;
            switch (value?.Trim().ToLowerInvariant())
            {
                case "fp4": activations = Nvfp4PromptActivations.Fp4; return true;
                case "q8": activations = Nvfp4PromptActivations.Q8; return true;
                default: return false;
            }
        }

        /// <summary>The option's spelling of <paramref name="value"/>.</summary>
        public static string Spell(Nvfp4PromptActivations value) => value == Nvfp4PromptActivations.Q8 ? "q8" : "fp4";

        /// <summary>The message a value outside <c>fp4|q8</c> gets.</summary>
        public static string Refusal(string value) => $"Unknown {OptionName} value '{value}'. Valid: fp4, q8.";

        /// <summary>
        /// A choice for the models constructed on the calling test's flow only (disposing restores the flow's previous one),
        /// unaffected by tests in parallel that set the process-wide choice.
        /// </summary>
        internal static IDisposable ForCurrentFlowInTests(Nvfp4PromptActivations value)
        {
            Nvfp4PromptActivations? previous = s_flow.Value;
            s_flow.Value = value;
            return new FlowScope(previous);
        }

        private sealed class FlowScope : IDisposable
        {
            private readonly Nvfp4PromptActivations? _previous;
            public FlowScope(Nvfp4PromptActivations? previous) => _previous = previous;
            public void Dispose() => s_flow.Value = _previous;
        }
    }
}
