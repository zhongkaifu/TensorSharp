// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.

using System;
using System.Linq;
using System.Reflection;
using TensorSharp.GGML;

namespace InferenceWeb.Tests;

/// <summary>
/// Guards the marshalling contract of the tensor-parallel "plan slot".
///
/// Several fused native kernels can either RUN their graph or, under tensor
/// parallelism, build it and hand back a plan for the caller to execute one per
/// rank. The native side decides which by asking whether <c>tp_plan_out</c> is a
/// null pointer — see <c>fused_matmul_quant_add_f32_impl</c> in
/// ggml_ops_fused.cpp and its siblings.
///
/// Declaring that parameter as <c>out IntPtr</c> silently breaks the contract:
/// the runtime passes the address of a stack local, so the native side sees a
/// NON-null slot on every call, including from callers that are not doing
/// tensor parallelism. Those callers then land in plan mode — the graph is
/// built, parked, and never executed — while the entry point still returns
/// success. The op computes nothing and reports that it worked.
///
/// That is exactly what made Nemotron's Mamba2 residual add a no-op under
/// --tp 2: `TryLinearAddInto` called FusedMatMulQuantAdd with the default
/// tpDegree=1 (it is a rank-0-only, replicated computation), the native gate
/// widened tpDegree to the process-wide TP degree, and all 23 Mamba2 layers -
/// 43% of the model - silently contributed nothing. The output stayed fluent
/// but was semantic garbage, which is the hardest possible failure to spot.
///
/// The fix is to declare the parameter as <c>IntPtr[]</c>, which marshals a
/// null array to a genuine null pointer. This test keeps it that way.
/// </summary>
public class GgmlTensorParallelPlanSlotContractTests
{
    private static Type NativeType =>
        typeof(GgmlBasicOps).Assembly.GetType("TensorSharp.GGML.GgmlNative")
        ?? throw new InvalidOperationException("TensorSharp.GGML.GgmlNative not found.");

    [Fact]
    public void EveryNativePlanSlotParameter_IsAnArray_SoNonTpCallersPassNull()
    {
        var offenders = NativeType
            .GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static)
            .SelectMany(m => m.GetParameters().Select(p => (Method: m, Param: p)))
            .Where(x => x.Param.Name == "tpPlanOut")
            .Where(x => x.Param.ParameterType != typeof(IntPtr[]))
            .Select(x => $"{x.Method.Name}({x.Param.ParameterType.Name} {x.Param.Name})")
            .OrderBy(s => s, StringComparer.Ordinal)
            .ToList();

        Assert.True(offenders.Count == 0,
            "A tensor-parallel plan slot must be declared 'IntPtr[]' so that a caller which is "
            + "NOT tensor-parallel marshals it to a null pointer. 'out IntPtr' always passes a "
            + "non-null slot, which drops those callers into plan mode: the native graph is built "
            + "but never executed and the call still reports success, so the op silently computes "
            + "nothing. Offending declarations:"
            + Environment.NewLine + string.Join(Environment.NewLine, offenders));
    }

    [Fact]
    public void PlanSlotParameters_Exist_SoThisGuardCannotSilentlyPass()
    {
        // If the parameter is ever renamed, the check above would vacuously pass.
        int count = NativeType
            .GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static)
            .SelectMany(m => m.GetParameters())
            .Count(p => p.Name == "tpPlanOut");

        Assert.True(count > 0,
            "No 'tpPlanOut' parameter found on TensorSharp.GGML.GgmlNative. If the tensor-parallel "
            + "plan slot was renamed, update EveryNativePlanSlotParameter_IsAnArray_SoNonTpCallersPassNull "
            + "to match, otherwise it silently guards nothing.");
    }
}
