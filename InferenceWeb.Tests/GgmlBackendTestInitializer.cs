// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Runtime.CompilerServices;
using TensorSharp.GGML;

namespace InferenceWeb.Tests;

/// <summary>
/// Pins the process-global GGML backend before any test runs.
///
/// The native bridge allows exactly ONE backend per process — the second
/// <c>GgmlNative.EnsureAvailable</c> with a different type throws
/// "Failed to initialize ggml-cpu. A different GGML backend was already
/// initialized in this process." Which backend wins is therefore decided by
/// whichever test happens to touch GGML first, and xunit gives no cross-class
/// ordering guarantee. That made the suite order-dependent: adding two
/// unrelated MLX test classes was enough to flip `GgmlCopyDtypeTests` and
/// `NemotronConvTests` (3 + 2 tests, all hard-coding
/// <see cref="GgmlBackendType.Cpu"/>) from green to red, purely because
/// something initialised a GPU backend earlier in the new ordering.
///
/// A module initializer runs before the first test in the assembly, so pinning
/// the backend here makes the choice deterministic instead of emergent. CPU is
/// the right default: it is the only backend guaranteed to exist on every host
/// the suite runs on (CI containers included), and every GGML test that does
/// NOT opt into a GPU backend is written against it.
///
/// Tests that genuinely need a GPU backend (the Wan parity suites) are opt-in
/// through their own env vars and return early when those are unset, so they
/// never reach GGML in a default run. When you DO opt into one of those, set
/// <c>TS_TEST_GGML_BACKEND</c> to the matching backend so this initializer
/// pins the same one rather than fighting it.
/// </summary>
internal static class GgmlBackendTestInitializer
{
    [ModuleInitializer]
    internal static void Initialize()
    {
        // This explicit fixture records production prompt/tokenizer behavior
        // without loading a native executor. The module initializer runs even
        // when vstest selects only that fixture, before its identity guard.
        if (Environment.GetEnvironmentVariable("TS_TEACHER_TOKEN_EXPORT") == "1")
            return;

        GgmlBackendType backend =
            (Environment.GetEnvironmentVariable("TS_TEST_GGML_BACKEND") ?? "cpu").Trim().ToLowerInvariant() switch
            {
                "metal" => GgmlBackendType.Metal,
                "cuda" => GgmlBackendType.Cuda,
                "vulkan" => GgmlBackendType.Vulkan,
                _ => GgmlBackendType.Cpu,
            };

        // Best effort: a host without the native bridge built must still be able
        // to run the ~1200 tests that never touch GGML. The tests that DO touch
        // it construct their own context and will surface the real error there.
        try
        {
            GgmlBasicOps.EnsureBackendAvailable(backend);
        }
        catch
        {
            // Intentionally swallowed — see above.
        }
    }
}
