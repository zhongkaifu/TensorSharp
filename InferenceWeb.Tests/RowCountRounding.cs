// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using TensorSharp.Cpu;
using TensorSharp.Models.Graph;

namespace InferenceWeb.Tests;

/// <summary>
/// What a test may ask of a CPU executor when it compares one computation run in steps of different row counts (a prompt in
/// pieces against whole, a continued prompt against a fresh one, a lane beside others against alone): DEVELOPMENT.md,
/// "Rounding by row count". No CPU executor gives a row the same bits whatever the rows beside it, and which of their row-count
/// switches a step meets depends on the host's ISA, so a bound measured on one host is not a bound: the PR lane's CPU jobs
/// failed run after run on tests that held across a row-count switch to what an AVX-512 host gave.
/// </summary>
internal static class RowCountRounding
{
    /// <summary>
    /// The fewest rows a managed-backend step needs for each of its rows to round as in any wider step, on every ISA: past the
    /// GEMM's skinny products (<see cref="CpuSgemm.SkinnyMaxM"/> rows or fewer sum in another order than a wider product's
    /// packed tile, with the AVX-512, AVX2 and portable kernels alike) and ARM64's row-by-row decode attention
    /// (<see cref="ManagedGraphExecutor.HostWindowRows"/> rows or fewer, against the tiled host kernel). Steps this wide
    /// give a prompt in pieces the whole prompt's bits on the managed backend. ggml-cpu has no such width: a 23-row piece of
    /// the dense Gemma 4 fixture rounds apart from the whole prompt there (4.9e-3 relL2).
    /// </summary>
    internal static readonly int ManagedWideRows = Math.Max(CpuSgemm.SkinnyMaxM, ManagedGraphExecutor.HostWindowRows) + 1;

    /// <summary>
    /// How far the Gemma 4 family's synthetic fixtures (Q4_0/Q8_0 random layers; the managed products quantize activations to
    /// 8 bits, so a one-ulp change can move a code by a step) carry a row-count switch: up to 4.9e-3 relL2 measured (the
    /// managed backend's AVX2 and portable kernels and the ARM64 runner, ggml-cpu on x64), where one cache row lost or written
    /// twice moves the logits by 3.2e-2 to 6.9e-2 and one changed prompt token a DiffusionGemma canvas by 9.9e-3 to 5.6e-2.
    /// </summary>
    internal const double GemmaFixtureNoise = 1e-2;
}
