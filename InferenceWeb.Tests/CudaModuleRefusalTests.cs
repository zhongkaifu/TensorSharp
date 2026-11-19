// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// A CUDA kernel module the allocator refuses (a stale or missing image, or no code for this GPU: the FP4 products on
// sm_103) used to drop its reason, so every route that wanted it fell back without a word. These tests pin the loader
// and the warn-once latch without a GPU: the factory runs once, its reason reaches the warning once, and the warning
// names the module, the GPU, the reason and what runs instead.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp.Cuda;
using Xunit;

namespace InferenceWeb.Tests;

public sealed class CudaModuleRefusalTests
{
    private sealed class Module { }

    private sealed class Lines
    {
        public List<string> Written { get; } = new();
        public void Write(string line)
        {
            lock (Written) Written.Add(line);
        }
    }

    [Fact]
    public void Warn_NamesModuleGpuReasonAndConsequence_OncePerModuleAndReason()
    {
        var lines = new Lines();
        var refusals = new CudaModuleRefusals(lines.Write);

        Assert.True(refusals.Warn("tensorsharp_fp4_kernels", 0, "no FP4 tensor-core code for this GPU.", "products run elsewhere"));
        Assert.False(refusals.Warn("tensorsharp_fp4_kernels", 1, "no FP4 tensor-core code for this GPU", "products run elsewhere"));
        string line = Assert.Single(lines.Written);
        Assert.StartsWith("WARNING: [cuda] tensorsharp_fp4_kernels is unavailable on GPU 0 (no FP4 tensor-core code for this GPU)", line);
        Assert.Contains("; products run elsewhere.", line);

        // Another reason, or another module, is news.
        Assert.True(refusals.Warn("tensorsharp_fp4_kernels", 1, "module version 3, expected 4", "products run elsewhere"));
        Assert.True(refusals.Warn("tensorsharp_flash_prefill", 0, "module version 3, expected 4", "attention runs elsewhere"));
        Assert.Equal(3, lines.Written.Count);
        Assert.False(refusals.Warn("tensorsharp_flash_prefill", 0, "module version 3, expected 4", "attention runs elsewhere"));
        Assert.Equal(3, lines.Written.Count);
    }

    [Fact]
    public void Warn_WithoutAReason_SaysSo_AndAFailingWriterDoesNotThrow()
    {
        var lines = new Lines();
        Assert.True(new CudaModuleRefusals(lines.Write).Warn("m", 2, null, "c"));
        Assert.Contains("(no reason given)", Assert.Single(lines.Written));

        var broken = new CudaModuleRefusals(_ => throw new InvalidOperationException("stderr closed"));
        Assert.True(broken.Warn("m", 0, "r", "c"));
        Assert.False(broken.Warn("m", 0, "r", "c"));
    }

    [Fact]
    public void LoadOnce_RefusedModule_CreatesOnceAndHandsTheReasonOverOnce()
    {
        object sync = new();
        Module module = null;
        int state = 0, creates = 0, prepares = 0;
        string refusal = null;
        var reasons = new List<string>();
        bool Refuse(out Module m, out string reason)
        {
            creates++;
            m = null;
            reason = "the loaded image is stale";
            return false;
        }

        for (int i = 0; i < 3; i++)
            Assert.Null(CudaModuleRefusals.LoadOnce(sync, ref module, ref state, ref refusal, () => prepares++, Refuse, reasons.Add));

        Assert.Equal((1, 1, 2), (creates, prepares, state));
        Assert.Equal("the loaded image is stale", refusal);
        Assert.Equal(new[] { "the loaded image is stale" }, reasons);
    }

    [Fact]
    public void LoadOnce_LoadedModule_IsReturnedWithoutAWarning()
    {
        object sync = new();
        Module module = null;
        int state = 0, creates = 0;
        string refusal = null;
        var reasons = new List<string>();
        var made = new Module();
        bool Load(out Module m, out string reason)
        {
            creates++;
            m = made;
            reason = null;
            return true;
        }

        Assert.Same(made, CudaModuleRefusals.LoadOnce(sync, ref module, ref state, ref refusal, null, Load, reasons.Add));
        Assert.Same(made, CudaModuleRefusals.LoadOnce(sync, ref module, ref state, ref refusal, null, Load, reasons.Add));
        Assert.Equal((1, 1), (creates, state));
        Assert.Null(refusal);
        Assert.Empty(reasons);
    }

    [Fact]
    public void LoadOnce_RefusalWithoutAReason_StillReportsOne()
    {
        object sync = new();
        Module module = null;
        int state = 0;
        string refusal = null, reported = null;
        bool Refuse(out Module m, out string reason)
        {
            m = null;
            reason = " ";
            return false;
        }

        Assert.Null(CudaModuleRefusals.LoadOnce(sync, ref module, ref state, ref refusal, null, Refuse, r => reported = r));
        Assert.Equal("no reason given", reported);
        Assert.Equal("no reason given", refusal);
    }

    private sealed class Slot
    {
        public Module Module;
        public int State;
        public string Refusal;
    }

    [Fact]
    public void LoadOnce_ConcurrentFirstUses_CreateAndReportOnce()
    {
        var slot = new Slot();
        int creates = 0, reports = 0;
        using var gate = new Barrier(8);
        bool Refuse(out Module m, out string reason)
        {
            Interlocked.Increment(ref creates);
            m = null;
            reason = "no FP4 tensor-core code for this GPU";
            return false;
        }

        Parallel.For(0, 8, new ParallelOptions { MaxDegreeOfParallelism = 8 }, _ =>
        {
            gate.SignalAndWait();
            Assert.Null(CudaModuleRefusals.LoadOnce(slot, ref slot.Module, ref slot.State, ref slot.Refusal, null, Refuse,
                _ => Interlocked.Increment(ref reports)));
        });

        Assert.Equal((1, 1, 2), (creates, reports, slot.State));
    }

    [Theory]
    [InlineData("tensorsharp_flash_prefill.ptx", "tensorsharp_flash_prefill")]
    [InlineData("tensorsharp_fp4_kernels", "tensorsharp_fp4_kernels")]
    public void ModuleName_DropsThePtxExtension(string file, string name)
        => Assert.Equal(name, CudaModuleRefusals.ModuleName(file));

    [Fact]
    public void Consequences_SayWhatRunsInstead()
    {
        // The three modules the allocator loads lazily each name their fallback, as the warning's second half.
        Assert.Contains("dequantizes each weight to F16 for cuBLAS", CudaModuleRefusals.Fp4Consequence);
        Assert.Contains("per-op path instead of the flash kernels", CudaModuleRefusals.FlashPrefillConsequence);
        Assert.Contains("four-row kernel", CudaModuleRefusals.SsmPrefillConsequence);
        Assert.Equal("tensorsharp_fp4_kernels", CudaModuleRefusals.ModuleName(CudaFp4Kernels.PtxFileName));
        Assert.Equal("tensorsharp_flash_prefill", CudaModuleRefusals.ModuleName(CudaFlashPrefillKernels.PtxFileName));
        Assert.Equal("tensorsharp_ssm_prefill", CudaModuleRefusals.ModuleName(CudaSsmPrefillKernels.PtxFileName));
    }
}
