using System;
using System.Collections.Generic;
using TensorSharp;
using TensorSharp.Cuda;

namespace TensorSharp.Models
{
    public partial class Qwen35Model
    {
        // Host top-k grouping plus device gather/scatter is the direct-CUDA
        // counterpart of ggml_mul_mat_id: each active expert sees one row batch,
        // so its quantized weights are streamed once for all assigned tokens.
        // Set to 0 only for A/B diagnostics.
        private static readonly bool s_qwenCudaMoeGroupedPrefill =
            Environment.GetEnvironmentVariable("TS_CUDA_MOE_PREFILL_GROUPED") != "0";

        private unsafe Tensor TryCudaMoEForwardPrefillGrouped(
            Tensor input, float* routerData, bool routeRowsAreLogits, int layer, int seqLen)
        {
            if (!s_qwenCudaMoeGroupedPrefill
                || _backend != BackendType.Cuda
                || seqLen <= 1
                || input?.Storage is not CudaStorage
                || !CanUseQwenCudaMoEOnDevice(layer))
            {
                return null;
            }

            int hiddenSize = Config.HiddenSize;
            var expertBatches = new List<(int token, float weight)>[_numExperts];
            for (int e = 0; e < _numExperts; e++)
                expertBatches[e] = new List<(int, float)>();

            var topExperts = new int[_numExpertsUsed];
            var routeWeights = new float[_numExpertsUsed];
            for (int token = 0; token < seqLen; token++)
            {
                SelectTopKRouteWeights(
                    routerData + (long)token * _numExperts,
                    routeRowsAreLogits, topExperts, routeWeights);
                for (int k = 0; k < _numExpertsUsed; k++)
                    expertBatches[topExperts[k]].Add((token, routeWeights[k]));
            }

            Tensor output = null;
            try
            {
                output = new Tensor(_allocator, DType.Float32, seqLen, hiddenSize);
                Ops.Fill(output, 0f);

                for (int expert = 0; expert < _numExperts; expert++)
                {
                    List<(int token, float weight)> batch = expertBatches[expert];
                    int batchSize = batch.Count;
                    if (batchSize == 0)
                        continue;

                    var rows = new int[batchSize];
                    var weights = new float[batchSize];
                    for (int i = 0; i < batchSize; i++)
                    {
                        rows[i] = batch[i].token;
                        weights[i] = batch[i].weight;
                    }

                    using Tensor rowIndices = CreateIntTensor(rows, batchSize);
                    using Tensor weightTensor = CreateFloatTensor(weights, batchSize);
                    using Tensor batchInput = Ops.IndexSelect(null, input, rowIndices);

                    Tensor gate = ExpertLinearForwardAlloc(batchInput, layer, expert, kind: 0);
                    Tensor up = ExpertLinearForwardAlloc(batchInput, layer, expert, kind: 1);
                    if (gate == null || up == null)
                    {
                        gate?.Dispose();
                        up?.Dispose();
                        output.Dispose();
                        return null;
                    }

                    Tensor down = null;
                    try
                    {
                        Ops.SiLUMul(gate, gate, up);
                        down = ExpertLinearForwardAlloc(gate, layer, expert, kind: 2);
                        if (down == null
                            || !CudaFusedOps.TryMoEScatterAddWeightedRows(
                                output, down, rowIndices, weightTensor))
                        {
                            output.Dispose();
                            return null;
                        }
                    }
                    finally
                    {
                        down?.Dispose();
                        up.Dispose();
                        gate.Dispose();
                    }
                }

                if (_hasSharedExperts != null
                    && layer < _hasSharedExperts.Length
                    && _hasSharedExperts[layer])
                {
                    Tensor sharedGate = null;
                    Tensor sharedUp = null;
                    Tensor sharedDown = null;
                    try
                    {
                        sharedGate = LinearForwardCached(
                            input, _ffnGateShexpQW[layer], _ffnGateShexpF32[layer]);
                        sharedUp = LinearForwardCached(
                            input, _ffnUpShexpQW[layer], _ffnUpShexpF32[layer]);
                        if (sharedGate == null || sharedUp == null)
                        {
                            output.Dispose();
                            return null;
                        }

                        Ops.SiLUMul(sharedGate, sharedGate, sharedUp);
                        sharedDown = LinearForwardCached(
                            sharedGate, _ffnDownShexpQW[layer], _ffnDownShexpF32[layer]);
                        Tensor gateVector = _hasSharedExpertGate != null
                            && layer < _hasSharedExpertGate.Length
                            && _hasSharedExpertGate[layer]
                            ? _ffnGateInpShexpVec?[layer]
                            : null;
                        if (sharedDown == null
                            || !CudaFusedOps.TryMoEAddSharedExpertBatched(
                                output, sharedDown, input, gateVector))
                        {
                            output.Dispose();
                            return null;
                        }
                    }
                    finally
                    {
                        sharedDown?.Dispose();
                        sharedUp?.Dispose();
                        sharedGate?.Dispose();
                    }
                }

                InvalidateTensorDeviceCache(output);
                return output;
            }
            catch (InvalidOperationException)
            {
                output?.Dispose();
                return null;
            }
            catch (NotSupportedException)
            {
                output?.Dispose();
                return null;
            }
        }
    }
}
