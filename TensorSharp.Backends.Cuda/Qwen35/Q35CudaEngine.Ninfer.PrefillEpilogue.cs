// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;

namespace TensorSharp.Cuda;

public sealed unsafe partial class Q35CudaEngine
{
    /// <summary>TS_Q35_NINFER_PREFILL_EPILOGUE=0 restores the cp.async GEMM tiles and the
    /// separate BF16 round, SwiGLU and quantize passes after the NInfer prefill projections.
    /// Unset or 1 runs them on the TMA tiles with the boundaries fused into the epilogue
    /// when the main CUDA image has the entries (same arithmetic, identical outputs).</summary>
    internal const string NinferPrefillEpilogueEnvironment = "TS_Q35_NINFER_PREFILL_EPILOGUE";
    private int _ninferPrefillEpilogueState; // 0 unresolved, 1 on, -1 off
    public long NinferPrefillEpilogueSubmissions => _k?.NinferPrefillEpilogueSubmissions ?? 0;

    private bool NinferPrefillEpilogueEnabled
    {
        get
        {
            if (_ninferPrefillEpilogueState == 0)
            {
                string raw = Environment.GetEnvironmentVariable(NinferPrefillEpilogueEnvironment);
                if (raw != null && raw != "0" && raw != "1")
                    throw new ArgumentException(NinferPrefillEpilogueEnvironment + " accepts only 0 or 1.");
                bool on = raw != "0" && _m.Ninfer != null && _k.NinferPrefillEpilogueAvailable;
                if (raw == "1" && !on)
                    throw new NotSupportedException(NinferPrefillEpilogueEnvironment + "=1 requires an NInfer target and a main CUDA image with the prefill epilogue entries.");
                _ninferPrefillEpilogueState = on ? 1 : -1;
                Console.Error.WriteLine($"[q35-cuda] NInfer prefill GEMM epilogues (BF16 round / SwiGLU / down NVFP4 quantization fused): {(on ? "on" : "off")}");
            }
            return _ninferPrefillEpilogueState > 0;
        }
    }

    private bool UsesNinferPrefillEpilogue(int rows, bool isVerify)
        => !isVerify && rows >= Q35Kernels.NinferPrefillEpilogueMinRows && NinferPrefillEpilogueEnabled;

    private IntPtr _ninferDownCodes, _ninferDownScales, _ninferDownRowScale;

    // The A4 matrix (and row count) whose NVFP4 input the last prefill norm already wrote
    // into _fp4c/_fp4s/_fp4r from _xn; the next quantization of _xn for it is skipped.
    // Every quantization into those planes consumes or clears it.
    private NinferMatrix _ninferPrequantizedFor;
    private int _ninferPrequantizedRows;

    /// <summary>NVFP4 input of an A4 projection into _fp4c/_fp4s/_fp4r, unless the preceding
    /// prefill norm already produced exactly it.</summary>
    private void QuantizeNinferInput(IntPtr input, NinferMatrix matrix, int rows)
    {
        var d = matrix.Descriptor;
        bool ready = input == _xn && ReferenceEquals(_ninferPrequantizedFor, matrix) && _ninferPrequantizedRows == rows;
        _ninferPrequantizedFor = null;
        if (!ready)
            _ninfer.QuantizeNvfp4(input, _fp4c, _fp4s, _fp4r, d.Columns, rows, d.InputScaleDivisor, Stream);
    }

    /// <summary>Ordinary prefill: the residual add + norm also writes the next A4 projection's
    /// NVFP4 input (ts_ni_add_norm_quant: the arithmetic of ts_ni_add_norm followed by
    /// ts_ni_quant_nvfp4, without re-reading the norm output). RTX 5090 32K prefill: 292 ms of
    /// add_norm + quantize became 108 + 48 ms.</summary>
    private bool TryNormalizeNinferPrefill(NinferResidualState state, IntPtr weight, int rows,
        bool candidatePass, NinferMatrix quantizeFor)
    {
        if (state != NinferResidualState.LegacyDelta || candidatePass || quantizeFor == null || _m.NEmbd % 256 != 0
            || !UsesNinferPrefillEpilogue(rows, false) || !NinferA4Plane(quantizeFor) || quantizeFor.Descriptor.Columns != _m.NEmbd)
            return false;
        _ninfer.AddRmsNormQuantize(_x, _block, weight, _xn, rows, _m.NEmbd, _m.Eps, _fp4c, _fp4s, _fp4r,
            quantizeFor.Descriptor.InputScaleDivisor, Stream);
        _ninferPrequantizedFor = quantizeFor;
        _ninferPrequantizedRows = rows;
        return true;
    }

    private bool NinferA4Plane(NinferMatrix matrix)
        => matrix != null && matrix.Descriptor.Format == CudaNinferProjectionFormat.NvFp4Planes
            && matrix.Descriptor.AllowA4 && !_ninferFiResident.ContainsKey(matrix)
            && Q35Kernels.NinferPrefillEpilogueFits(matrix.Data, matrix.Scales, matrix.Descriptor.Rows, matrix.Descriptor.Columns);

    /// <summary>gate|up projection whose epilogue emits the down projection's NVFP4
    /// input directly: the activation ts_ni_swiglu would store, quantized exactly as
    /// ts_ni_quant_nvfp4 would quantize it with the down matrix's input divisor.</summary>
    private bool TryNinferPrefillGateUpSwiGluQuant(NinferMatrix gateUp, NinferMatrix down, int rows, bool isVerify)
    {
        if (!UsesNinferPrefillEpilogue(rows, isVerify) || !NinferA4Plane(gateUp) || !NinferA4Plane(down)
            || gateUp.Descriptor.Rows != 2 * _m.NFf || down.Descriptor.Columns != _m.NFf || _m.NFf % 256 != 0)
            return false;
        if (_ninferDownCodes == IntPtr.Zero)
        {
            _ninferDownCodes = Alloc((long)RowCap * _m.NFf / 2);
            _ninferDownScales = Alloc((long)RowCap * _m.NFf / 16);
            _ninferDownRowScale = Alloc((long)RowCap * 4);
        }
        var g = gateUp.Descriptor;
        QuantizeNinferInput(_xn, gateUp, rows);
        _k.GemmW4A4NinferSwiGluQuant(gateUp.Data, gateUp.Scales, g.Rows, g.Columns, _fp4c, _fp4s, _fp4r, rows,
            1.0f / (g.InputScaleDivisor * g.WeightScaleDivisor), rows > 128 && rows != 1024,
            _ninferDownCodes, _ninferDownScales, _ninferDownRowScale, down.Descriptor.InputScaleDivisor, Stream);
        return true;
    }

    /// <summary>The down projection of TryNinferPrefillGateUpSwiGluQuant's quantized input,
    /// into the residual delta exactly as ProjectNinferResidual's unfused prefill route.</summary>
    private NinferResidualState ProjectNinferPrequantizedDown(NinferMatrix down, int rows)
    {
        var d = down.Descriptor;
        _k.GemmW4A4NinferPrefill(down.Data, down.Scales, d.Rows, d.Columns, _ninferDownCodes, _ninferDownScales,
            _ninferDownRowScale, rows, 1.0f / (d.InputScaleDivisor * d.WeightScaleDivisor), _block,
            Q35Kernels.NinferPrefillEpilogue.Plain, Stream);
        return NinferProjectedResidualState(false);
    }

    /// <summary>gate|up projection straight to the FFN activation for ordinary prefill
    /// rows. Reproduces NinferProject (with its rows &gt; 128 &amp;&amp; rows != 1024 BF16
    /// round) followed by ts_ni_swiglu, without the two extra passes.</summary>
    private bool TryNinferPrefillGateUpSwiGlu(NinferMatrix matrix, int rows, bool isVerify)
    {
        if (!UsesNinferPrefillEpilogue(rows, isVerify) || !NinferA4Plane(matrix) || matrix.Descriptor.Rows != 2 * _m.NFf)
            return false;
        var d = matrix.Descriptor;
        QuantizeNinferInput(_xn, matrix, rows);
        float alpha = 1.0f / (d.InputScaleDivisor * d.WeightScaleDivisor);
        var epilogue = rows > 128 && rows != 1024
            ? Q35Kernels.NinferPrefillEpilogue.SwiGluRoundedInputs
            : Q35Kernels.NinferPrefillEpilogue.SwiGlu;
        _k.GemmW4A4NinferPrefill(matrix.Data, matrix.Scales, d.Rows, d.Columns, _fp4c, _fp4s, _fp4r,
            rows, alpha, _act, epilogue, Stream);
        return true;
    }
}
