// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ---------------------------------------------------------------------------
// ggml-cuda keeps a CUDA graph for each run of nodes it is handed, keyed by the
// run's first node. A caller that captures a whole fused-op backend run itself
// (tsg_inkling_compute) switches ggml-cuda's own capture off for the runs
// inside it, so ggml-cuda evaluates their nodes into the outer capture instead
// of beginning a capture of its own on the capturing stream.
//
// GGML_CUDA_USE_GRAPHS is defined for ggml-cuda's own directory only; the
// CMakeLists defines it for this file as well, so the CUDA context and its
// graph table have ggml-cuda's layout here (the other TensorSharp CUDA files
// read only the context's leading members). The vendored ggml is used through
// its headers only.
// ---------------------------------------------------------------------------
#include "tsg_cuda_graph_keys.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/graph.cuh"
#include "ggml-cuda/mmvq.cuh"
#include "ggml-cuda/mmvf.cuh"

// ggml-cuda's kernel choices for a MUL_MAT_ID (declared in mmq.cuh and mmf.cuh, whose kernel templates this file does not need).
bool ggml_cuda_should_use_mmq(enum ggml_type type, int cc, int64_t ne11, int64_t n_experts);
bool ggml_cuda_should_use_mmf(enum ggml_type type, int cc, int warp_size, const int64_t * scr0_ne, const size_t * src0_nb,
                              const int src1_ncols, bool mul_mat_id);

namespace
{
    // ggml-cuda.cu's ggml_cuda_mul_mat_id_needs_sync (static there): whether a MUL_MAT_ID takes the fallback that copies the
    // expert ids to the host and synchronizes the stream. Keep it in step with ExternalProjects/ggml.
    bool mul_mat_id_needs_sync(const ggml_tensor * dst, int cc, int warp_size)
    {
        const ggml_tensor * src0 = dst->src[0];
        const ggml_tensor * src1 = dst->src[1];
        if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) return true;
        if (dst->ne[2] <= MMVQ_MAX_BATCH_SIZE)
        {
            if (ggml_is_quantized(src0->type))
            {
                if (dst->ne[2] <= get_mmvq_mmid_max_batch(src0->type, cc)) return false;
            }
            else if (src0->type == GGML_TYPE_F8_E4M3 && ggml_cuda_should_use_mmvf(src0, cc, warp_size, dst->ne[2]))
                return false;
            else if (src0->type != GGML_TYPE_F8_E4M3 && GGML_CUDA_CC_IS_AMD(cc))
                return false;
        }
        if (ggml_cuda_should_use_mmq(src0->type, cc, src1->ne[2], /*n_experts=*/src0->ne[2])) return false;
        if (ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, src1->ne[2], /*mul_mat_id=*/true)) return false;
        return true;
    }
}

bool tsg_cuda_graph_capturable(ggml_backend_t cuda_backend, const ggml_cgraph * graph)
{
    auto * context = static_cast<ggml_backend_cuda_context *>(cuda_backend->context);
    const int cc = ggml_cuda_info().devices[context->device].cc;
    const int warp_size = ggml_cuda_info().devices[context->device].warp_size;
    for (int i = 0; i < graph->n_nodes; i++)
        if (graph->nodes[i]->op == GGML_OP_MUL_MAT_ID && mul_mat_id_needs_sync(graph->nodes[i], cc, warp_size))
            return false;
    return true;
}

void tsg_cuda_disable_graphs(ggml_backend_t cuda_backend, const void * const * keys, int count)
{
#ifdef USE_CUDA_GRAPH
    auto * context = static_cast<ggml_backend_cuda_context *>(cuda_backend->context);
    // The table's one sticky per-graph switch: ggml-cuda sets it for GPUs without graph support and never clears it.
    for (int i = 0; i < count; i++)
        context->cuda_graph(keys[i])->disable_due_to_gpu_arch = true;
#else
    (void)cuda_backend; (void)keys; (void)count;
#endif
}
