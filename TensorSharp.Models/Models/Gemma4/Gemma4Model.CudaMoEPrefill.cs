using System;
using System.Collections.Generic;
using TensorSharp;
using TensorSharp.Cuda;

namespace TensorSharp.Models
{
    public partial class Gemma4Model
    {
        // Direct-CUDA grouped MoE prefill. Host routing is already available in
        // Gemma's reference path; only the large activation gather/scatter moves
        // to the device. Set to 0 for an exact legacy-path A/B comparison.
        private static readonly bool s_gemmaCudaMoeGroupedPrefill =
            Environment.GetEnvironmentVariable("TS_CUDA_MOE_PREFILL_GROUPED") != "0";

        private bool TryMoEGroupedCudaPrefill(
            Tensor moeInput,
            Tensor output,
            int[] selectedExperts,
            float[] routingWeights,
            int layer,
            string prefix,
            int seqLen,
            int hiddenDim)
        {
            if (!s_gemmaCudaMoeGroupedPrefill
                || _backend != BackendType.Cuda
                || seqLen <= 1
                || moeInput?.Storage is not CudaStorage
                || output?.Storage is not CudaStorage
                || !CanUseCudaMoEOnDevice(layer)
                || selectedExperts == null
                || routingWeights == null
                || selectedExperts.Length < seqLen * _numExpertsUsed
                || routingWeights.Length < seqLen * _numExpertsUsed)
            {
                return false;
            }

            var expertBatches = new List<(int token, float weight)>[_numExperts];
            for (int expert = 0; expert < _numExperts; expert++)
                expertBatches[expert] = new List<(int, float)>();

            float[] perExpertScale = GetMoEPerExpertScale(layer, prefix);
            for (int token = 0; token < seqLen; token++)
            {
                int baseRoute = token * _numExpertsUsed;
                for (int k = 0; k < _numExpertsUsed; k++)
                {
                    int route = baseRoute + k;
                    int expert = selectedExperts[route];
                    if ((uint)expert >= (uint)_numExperts)
                        return false;
                    float weight = routingWeights[route];
                    if (perExpertScale != null)
                        weight *= perExpertScale[expert];
                    expertBatches[expert].Add((token, weight));
                }
            }

            try
            {
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
                    using Tensor batchInput = Ops.IndexSelect(null, moeInput, rowIndices);

                    string fusedKey = $"{prefix}.ffn_gate_up_exps.{expert}.weight";
                    string downKey = $"{prefix}.ffn_down_exps.{expert}.weight";
                    Tensor expertOut = null;
                    try
                    {
                        if (_weights.ContainsKey(fusedKey) || _quantWeights.ContainsKey(fusedKey))
                        {
                            expertOut = FFNGelu(batchInput, fusedKey, downKey, batchSize);
                        }
                        else
                        {
                            string gateKey = $"{prefix}.ffn_gate_exps.{expert}.weight";
                            string upKey = $"{prefix}.ffn_up_exps.{expert}.weight";
                            if (!(_weights.ContainsKey(gateKey) || _quantWeights.ContainsKey(gateKey))
                                || !(_weights.ContainsKey(upKey) || _quantWeights.ContainsKey(upKey))
                                || !(_weights.ContainsKey(downKey) || _quantWeights.ContainsKey(downKey)))
                            {
                                return false;
                            }
                            using Tensor gate = LinearForward(batchInput, gateKey);
                            using Tensor up = LinearForward(batchInput, upKey);
                            Ops.GELUMul(gate, gate, up);
                            expertOut = LinearForward(gate, downKey);
                        }

                        if (expertOut == null
                            || !CudaFusedOps.TryMoEScatterAddWeightedRows(
                                output, expertOut, rowIndices, weightTensor))
                        {
                            return false;
                        }
                    }
                    finally
                    {
                        expertOut?.Dispose();
                    }
                }

                InvalidateTensorDeviceCache(output);
                return true;
            }
            catch (InvalidOperationException)
            {
                return false;
            }
            catch (NotSupportedException)
            {
                return false;
            }
        }
    }
}
