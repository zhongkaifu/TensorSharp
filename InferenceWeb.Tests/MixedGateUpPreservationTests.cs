// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

public sealed class MixedGateUpPreservationTests
{
    [GgmlFact(BackendType.GgmlCpu)]
    public void CpuPreservesMixedWeightsAndStillFusesMatchingFormats() => Verify(BackendType.GgmlCpu);

    [GgmlFact(BackendType.GgmlCuda)]
    public void CudaPreservesMixedWeightsAndStillFusesMatchingFormats() => Verify(BackendType.GgmlCuda);

    private static void Verify(BackendType backend)
    {
        const string gate = "blk.0.ffn_gate.weight", up = "blk.0.ffn_up.weight";
        var q8 = SyntheticGguf.Gen(gate, 0.1f, 256, 4);
        q8.Type = SyntheticGguf.GgmlType.Q8_0;
        var half = SyntheticGguf.Gen(up, 0.1f, 256, 4);
        half.Type = SyntheticGguf.GgmlType.F16;
        Check(q8, half, expectSplit: true);
        // UD formats that the former loader would dequantize and requantize.
        Check(SyntheticGguf.Blocks(gate, SyntheticGguf.GgmlType.IQ4_XS, 0.1f, 256, 4),
            SyntheticGguf.Blocks(up, SyntheticGguf.GgmlType.Q6_K, 0.1f, 256, 4), expectSplit: true);
        half.Type = SyntheticGguf.GgmlType.Q8_0;
        Check(q8, half, expectSplit: false);

        void Check(SyntheticGguf.Tensor gateTensor, SyntheticGguf.Tensor upTensor, bool expectSplit)
        {
            string path = Path.Combine(Path.GetTempPath(), $"ts-mixed-gate-up-{Guid.NewGuid():N}.gguf");
            string previous = Environment.GetEnvironmentVariable("TS_WEIGHT_FUSION_COPIES");
            Environment.SetEnvironmentVariable("TS_WEIGHT_FUSION_COPIES", "1");
            try
            {
                SyntheticGguf.Write(path,
                    [new SyntheticGguf.Str { Key = "general.architecture", V = "probe" }],
                    [gateTensor, upTensor]);
                using var model = new FusionModel(path, backend);
                var originalGate = model.Weight(gate);
                var originalUp = model.Weight(up);
                model.Fuse();
                if (expectSplit)
                {
                    Assert.Null(model.Weight("blk.0.ffn_gate_up.weight"));
                    Assert.Same(originalGate, model.Weight(gate));
                    Assert.Same(originalUp, model.Weight(up));
                    Assert.Equal((int)gateTensor.Type, originalGate.GgmlType);
                    Assert.Equal((int)upTensor.Type, originalUp.GgmlType);
                    Assert.Equal(gateTensor.Raw(), Bytes(originalGate));
                    Assert.Equal(upTensor.Raw(), Bytes(originalUp));
                }
                else
                {
                    Assert.Null(model.Weight(gate));
                    Assert.Null(model.Weight(up));
                    var fused = model.Weight("blk.0.ffn_gate_up.weight");
                    Assert.NotNull(fused);
                    Assert.Equal((int)gateTensor.Type, fused.GgmlType);
                    Assert.Equal(gateTensor.Raw().Concat(upTensor.Raw()).ToArray(), Bytes(fused));
                }
            }
            finally
            {
                Environment.SetEnvironmentVariable("TS_WEIGHT_FUSION_COPIES", previous);
                File.Delete(path);
            }
        }
    }

    private static byte[] Bytes(QuantizedWeight weight)
    {
        var result = new byte[checked((int)weight.RawBytes)];
        Marshal.Copy(weight.Data, result, 0, result.Length);
        return result;
    }

    private sealed class FusionModel : ModelBase
    {
        public FusionModel(string path, BackendType backend) : base(path, backend)
        {
            try { LoadWeights(); }
            catch { Dispose(); throw; }
        }
        protected override bool SupportsSplitGateUpFfn => true;
        public void Fuse() => FuseGateUpWeights(1);
        public QuantizedWeight Weight(string name) => _quantWeights.GetValueOrDefault(name);
        protected override float[] ForwardCore(int[] tokens) => throw new NotSupportedException();
        protected override void ResetKVCacheCore() { }
    }
}
