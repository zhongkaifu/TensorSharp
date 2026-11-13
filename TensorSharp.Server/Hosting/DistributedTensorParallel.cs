// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using TensorSharp.Distributed;

namespace TensorSharp.Server.Hosting;

/// <summary>
/// The Server's <see cref="ModelService.TensorParallelGroupFactory"/>: builds the
/// on-node tensor-parallel group for a multi-node load from the
/// <c>TENSORSHARP_TP_*</c> variables that <c>--tp</c> / <c>--tp-node-id</c> /
/// <c>--tp-peers</c> translate into.
///
/// <para>
/// It lives in the Server rather than in TensorSharp.Chat because
/// TensorSharp.Distributed references TensorSharp.Backends.Cuda, and the chat
/// library has to stay linkable by hosts that cannot carry either (an iOS app).
/// Those hosts simply leave the factory unset and every load is single-node.
/// </para>
/// </summary>
public static class DistributedTensorParallel
{
    /// <summary>
    /// Null when <c>TENSORSHARP_TP_NODE_ID</c> / <c>TENSORSHARP_TP_PEERS</c> are not
    /// both set (single-node), otherwise the group for <paramref name="backend"/>.
    /// </summary>
    public static ITensorParallelGroup? CreateGroup(BackendType backend)
    {
        var options = ModelParallelismOptions.Parse(Array.Empty<string>());
        var distConfig = options.Distributed;
        if (distConfig == null)
            return null;
        if (backend is not (BackendType.Cuda or BackendType.GgmlCuda or BackendType.GgmlVulkan))
            throw new ArgumentException("Distributed --tp requires cuda, ggml_cuda, or ggml_vulkan.");

        if (options.IsDistributedPipeline)
        {
            // This node's --layer-split stages of --tp GPUs each; the nodes exchange their shapes and capacities.
            ITensorParallelGroup local = ModelBase.CreateLocalStageGroup(backend, options.LayerSplitDegree, options.TpDegree);
            return new DistributedLayerSplitGroup(local, distConfig.NodeId, distConfig.PeerEndpoints,
                options.TpDegree, ModelBase.MeasureStageCapacity(local, backend, options.TpDegree));
        }

        // The on-node group has to match the backend: direct CUDA
        // drives CudaAllocators, the ggml backends drive per-rank
        // ggml backends.
        return backend is BackendType.GgmlCuda or BackendType.GgmlVulkan
            ? new DistributedTensorParallelGroup(
                ModelBase.CreateGgmlLocalTpGroup(backend, distConfig.LocalDegree),
                distConfig.NodeId, distConfig.PeerEndpoints)
            : new DistributedTensorParallelGroup(
                distConfig.LocalDegree, distConfig.NodeId, distConfig.PeerEndpoints);
    }

}
