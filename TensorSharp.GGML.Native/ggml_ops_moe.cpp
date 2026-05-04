// MoE FFN prefill kernel using ggml_mul_mat_id.
//
// This kernel collapses an entire MoE FFN forward pass — gate + up projections,
// SwiGLU activation, down projection, expert weighting, and per-token expert
// aggregation — into ONE GGML graph dispatch per layer per ubatch, regardless
// of how many experts are active.
//
// llama.cpp's `build_moe_ffn` is the reference: it builds the same graph using
// 2-3 `ggml_mul_mat_id` ops per layer (one for gate_up, one for down, plus an
// optional split when gate_up is fused). TensorSharp's previous MoE prefill
// path issued ~3 GGML graph submissions per active expert per layer (~3000+
// dispatches per pp2048 forward on GPT-OSS). This kernel collapses that to
// O(1) dispatches per layer.
//
// Three activation modes are supported:
//   - SwiGLU split: out = silu(gate) * up         (Qwen35, generic MoE)
//   - SwiGLU OAI : out = clamp_silu(gate, alpha)  (GPT-OSS specific variant)
//                  * (clamp(up, ±limit) + 1)
//   - GEGLU split: out = gelu(gate) * up          (Gemma 4 MoE; tanh-approx
//                                                  GELU matching llama.cpp's
//                                                  GGML_GLU_OP_GEGLU)
//
// Two weight layouts are supported:
//   - Fused gate_up: gate weight has ne1 = 2*n_ff, up_data is null.
//   - Separate gate / up: each has ne1 = n_ff, up_data is non-null.
//
// All quantized weights are passed as a single contiguous block with a base
// pointer + total bytes. The expected layout is the per-expert quantized data
// stacked along the expert dimension (matching the original GGUF on-disk layout
// for `_exps.weight` tensors), so for mmap'd models there is zero extra copy
// cost — the C# layer simply passes the base pointer of expert 0.

#include "ggml_ops_internal.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <cstdint>
#include <cstring>
#include <vector>

using namespace tsg;

namespace
{
        // Bind a read-only weight tensor to its host data via the cacheable buffer
        // path when possible (so subsequent calls hit the cached backend buffer),
        // otherwise via the host-pointer mapped buffer (zero-copy on Metal /
        // unified memory), otherwise fall back to a pending CPU upload after the
        // backend buffer is allocated.
        struct WeightUploadInfo
        {
            ggml_tensor* tensor = nullptr;
            void* host_data = nullptr;
            std::size_t bytes = 0;
            bool needs_upload = false;
        };

        bool bind_readonly_weight_3d(
            ggml_context* ctx,
            ggml_backend_t backend,
            ggml_backend_dev_t dev,
            void* host_data,
            std::size_t bytes,
            ggml_tensor* tensor,
            std::vector<WeightUploadInfo>& uploads,
            std::vector<BufferHandle>& ephem_buffers)
        {
            if (tensor == nullptr || host_data == nullptr || bytes == 0)
                return false;

            if (dev != nullptr && bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr;
                void* addr = nullptr;
                bool needs_upload = false;
                if (try_get_cacheable_tensor_buffer(backend, dev, tensor, host_data, bytes, buf, addr, needs_upload))
                {
                    if (ggml_backend_tensor_alloc(buf, tensor, addr) == GGML_STATUS_SUCCESS)
                    {
                        if (needs_upload)
                        {
                            uploads.push_back({tensor, host_data, bytes, true});
                        }
                        return true;
                    }
                    invalidate_cached_buffer(host_data);
                }
            }

            if (bytes >= 4096)
            {
                ggml_backend_buffer_t buf = nullptr;
                if (try_get_host_ptr_buffer(backend, dev, host_data, bytes, false, buf))
                {
                    ephem_buffers.emplace_back(buf);
                    if (ggml_backend_tensor_alloc(buf, tensor, host_data) == GGML_STATUS_SUCCESS)
                    {
                        return true;
                    }
                }
            }

            // Fall back to deferred upload after backend buffer allocation.
            uploads.push_back({tensor, host_data, bytes, true});
            return true;
        }

        // Build the chain of adds that aggregates the per-expert outputs along
        // the n_used dimension. Equivalent to:
        //   for u = 0..n_used-1: out += weighted[:, u, :]
        // but expressed as a chain of ggml_add over views so the scheduler can
        // see it as a single graph.
        ggml_tensor* build_expert_aggregation(
            ggml_context* ctx,
            ggml_tensor* weighted,
            int hidden_dim,
            int n_used,
            int seq_len)
        {
            std::vector<ggml_tensor*> views(n_used);
            for (int u = 0; u < n_used; ++u)
            {
                views[u] = ggml_view_2d(
                    ctx,
                    weighted,
                    static_cast<std::int64_t>(hidden_dim),
                    static_cast<std::int64_t>(seq_len),
                    weighted->nb[2],
                    static_cast<std::size_t>(u) * weighted->nb[1]);
                if (views[u] == nullptr)
                    return nullptr;
            }
            ggml_tensor* acc = views[0];
            for (int u = 1; u < n_used; ++u)
            {
                acc = ggml_add(ctx, acc, views[u]);
                if (acc == nullptr)
                    return nullptr;
            }
            return acc;
        }

        // Activation modes for the fused MoE prefill kernel.
        enum MoEActivation : int
        {
            MOE_ACT_SWIGLU_SPLIT = 0, // silu(gate) * up
            MOE_ACT_SWIGLU_OAI   = 1, // gpt-oss clamped variant
            MOE_ACT_GEGLU_SPLIT  = 2, // gelu(gate) * up   (Gemma 4 MoE)
        };

    // Parameter bag for the unified MoE FFN graph builder. The two public
    // entry points (the original `MoEFFNPrefill` writing to `hidden_out` and
    // the Gemma 4 residual-fused variant writing through
    // `residual_in_out += rms_norm(moe_out, post_norm_w)`) only differ in the
    // graph node sequence after the per-expert aggregation, so the body is
    // shared and parameterised through this struct.
    struct MoEFFNGraphParams
    {
        // Common inputs
        float* hidden_in = nullptr;
        int seq_len = 0;
        int hidden_dim = 0;
        int n_ff = 0;
        int num_experts = 0;
        int n_used = 0;
        const std::int32_t* selected_experts = nullptr;
        const float* routing_weights = nullptr;

        void* gate_data = nullptr;
        int gate_type = 0;
        std::int64_t gate_ne0 = 0;
        std::int64_t gate_ne1 = 0;
        std::int64_t gate_total_bytes = 0;

        void* up_data = nullptr;
        int up_type = 0;
        std::int64_t up_ne0 = 0;
        std::int64_t up_ne1 = 0;
        std::int64_t up_total_bytes = 0;

        void* down_data = nullptr;
        int down_type = 0;
        std::int64_t down_ne0 = 0;
        std::int64_t down_ne1 = 0;
        std::int64_t down_total_bytes = 0;

        const float* gate_bias = nullptr;
        const float* up_bias = nullptr;
        const float* down_bias = nullptr;

        int activation_type = 0;
        float oai_alpha = 1.702f;
        float oai_limit = 7.0f;

        // Output mode A: write moe_out to a dedicated buffer (no residual fold).
        // Used by the standalone TSGgml_MoEFFNPrefillSwiGLUQuantF32.
        float* hidden_out = nullptr;

        // Output mode B (Gemma 4 residual variant): the kernel computes
        //   residual_in_out += rms_norm(moe_out, eps) * post_norm_w
        // entirely on-device. Both pointers must be non-null and post_norm_w
        // must point to a hidden_dim-element F32 weight vector. eps is the
        // RMSNorm epsilon (ModelConfig.Eps in C#).
        float* residual_in_out = nullptr;
        const float* post_norm_w = nullptr;
        float post_norm_eps = 1e-6f;
    };

    int moe_ffn_prefill_graph_impl(const MoEFFNGraphParams& p)
    {
        float* const hidden_in = p.hidden_in;
        const int seq_len = p.seq_len;
        const int hidden_dim = p.hidden_dim;
        const int n_ff = p.n_ff;
        const int num_experts = p.num_experts;
        const int n_used = p.n_used;
        const std::int32_t* const selected_experts = p.selected_experts;
        const float* const routing_weights = p.routing_weights;
        void* const gate_data = p.gate_data;
        const int gate_type = p.gate_type;
        const std::int64_t gate_ne0 = p.gate_ne0;
        const std::int64_t gate_ne1 = p.gate_ne1;
        const std::int64_t gate_total_bytes = p.gate_total_bytes;
        void* const up_data = p.up_data;
        const int up_type = p.up_type;
        const std::int64_t up_ne0 = p.up_ne0;
        const std::int64_t up_ne1 = p.up_ne1;
        const std::int64_t up_total_bytes = p.up_total_bytes;
        void* const down_data = p.down_data;
        const int down_type = p.down_type;
        const std::int64_t down_ne0 = p.down_ne0;
        const std::int64_t down_ne1 = p.down_ne1;
        const std::int64_t down_total_bytes = p.down_total_bytes;
        const float* const gate_bias = p.gate_bias;
        const float* const up_bias = p.up_bias;
        const float* const down_bias = p.down_bias;
        const int activation_type = p.activation_type;
        const float oai_alpha = p.oai_alpha;
        const float oai_limit = p.oai_limit;
        float* const hidden_out = p.hidden_out;
        float* const residual_in_out = p.residual_in_out;
        const float* const post_norm_w = p.post_norm_w;
        const float post_norm_eps = p.post_norm_eps;
        // Output mode B (residual fold) is selected when both the residual
        // buffer and the post-norm weight are supplied. Mutually exclusive
        // with mode A.
        const bool residual_mode = (residual_in_out != nullptr) && (post_norm_w != nullptr);
        const bool standalone_mode = (hidden_out != nullptr) && !residual_mode;
        if (!ensure_backend()) return 0;

        // --- Argument validation ---
        if (!residual_mode && !standalone_mode)
        {
            set_last_error("MoE prefill: caller must supply either hidden_out (standalone) or both residual_in_out and post_norm_w (residual mode).");
            return 0;
        }
        if (residual_mode && hidden_out != nullptr)
        {
            set_last_error("MoE prefill: residual_in_out and hidden_out are mutually exclusive output modes.");
            return 0;
        }
        if (hidden_in == nullptr || selected_experts == nullptr || routing_weights == nullptr)
        {
            set_last_error("MoE prefill: null host pointer for input/ids/weights.");
            return 0;
        }
        if (seq_len <= 0 || hidden_dim <= 0 || n_ff <= 0 || num_experts <= 0 || n_used <= 0)
        {
            set_last_error("MoE prefill: invalid shape parameter (must all be > 0).");
            return 0;
        }
        if (n_used > num_experts)
        {
            set_last_error("MoE prefill: n_used cannot exceed num_experts.");
            return 0;
        }
        if (residual_mode && !std::isfinite(post_norm_eps))
        {
            set_last_error("MoE prefill: post_norm_eps must be a finite positive value.");
            return 0;
        }

        const bool fused_gate_up = (up_data == nullptr);
        const std::int64_t expected_gate_ne1 = fused_gate_up
            ? static_cast<std::int64_t>(2) * n_ff
            : static_cast<std::int64_t>(n_ff);

        if (gate_data == nullptr ||
            gate_ne0 != static_cast<std::int64_t>(hidden_dim) ||
            gate_ne1 != expected_gate_ne1 ||
            gate_total_bytes <= 0)
        {
            set_last_error("MoE prefill: gate weight pointer/shape mismatch.");
            return 0;
        }
        if (!fused_gate_up &&
            (up_ne0 != static_cast<std::int64_t>(hidden_dim) ||
             up_ne1 != static_cast<std::int64_t>(n_ff) ||
             up_total_bytes <= 0))
        {
            set_last_error("MoE prefill: up weight pointer/shape mismatch.");
            return 0;
        }
        if (down_data == nullptr ||
            down_ne0 != static_cast<std::int64_t>(n_ff) ||
            down_ne1 != static_cast<std::int64_t>(hidden_dim) ||
            down_total_bytes <= 0)
        {
            set_last_error("MoE prefill: down weight pointer/shape mismatch.");
            return 0;
        }

        const std::size_t hidden_bytes =
            static_cast<std::size_t>(seq_len) *
            static_cast<std::size_t>(hidden_dim) * sizeof(float);

        // --- Context allocation ---
        // Conservative size to fit graph nodes + ephemeral tensor headers for
        // up to ~16 active experts in the aggregation chain.
        const std::size_t ctx_size = 4 * 1024 * 1024;
        PooledContextHandle context;
        if (!context.init(ctx_size))
        {
            set_last_error("MoE prefill: failed to acquire ggml context.");
            return 0;
        }
        ggml_context* ctx = context.value;

        ggml_backend_dev_t dev = ggml_backend_get_device(g_backend);

        // --- Tensor allocation ---
        // Hidden state tensors are 2D [hidden_dim, seq_len] (column-major as
        // ggml expects; ne0 is the innermost / leading dim and seq_len is the
        // outer dim, matching the row-major C# layout
        //   hidden[s * hidden_dim + d]
        // which equals ggml's hidden_t[d, s]).
        ggml_tensor* hidden_t = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F32, hidden_dim, seq_len);
        // In standalone mode this holds the MoE output. In residual mode the
        // residual buffer (passed in by the caller, holds the dense FFN output
        // we add the normed MoE result to) plays the same structural role.
        ggml_tensor* hidden_out_t = ggml_new_tensor_2d(
            ctx, GGML_TYPE_F32, hidden_dim, seq_len);

        // Residual-mode-only weight: F32 [hidden_dim] post_ffw_norm_2.weight,
        // multiplied with rms_norm(moe_out) before the residual add.
        ggml_tensor* post_norm_w_t = nullptr;
        if (residual_mode)
        {
            post_norm_w_t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, hidden_dim);
        }

        ggml_tensor* gate_w = ggml_new_tensor_3d(
            ctx, static_cast<ggml_type>(gate_type),
            gate_ne0, gate_ne1, num_experts);
        ggml_tensor* up_w = nullptr;
        if (!fused_gate_up)
        {
            up_w = ggml_new_tensor_3d(
                ctx, static_cast<ggml_type>(up_type),
                up_ne0, up_ne1, num_experts);
        }
        ggml_tensor* down_w = ggml_new_tensor_3d(
            ctx, static_cast<ggml_type>(down_type),
            down_ne0, down_ne1, num_experts);

        // ids: i32 [n_used, seq_len] — innermost dim n_used so memory layout
        // matches selected_experts[s * n_used + u].
        ggml_tensor* ids_t = ggml_new_tensor_2d(
            ctx, GGML_TYPE_I32, n_used, seq_len);

        // routing weights: [1, n_used, seq_len] — broadcasting scalar per
        // (token, used-expert) slot. Layout matches routing_weights[s * n_used + u].
        ggml_tensor* weights_t = ggml_new_tensor_3d(
            ctx, GGML_TYPE_F32, 1, n_used, seq_len);

        // Optional per-expert biases. ggml_add_id consumes a 2D [bias_dim, num_experts]
        // f32 tensor and adds bias[:, ids[u, t]] to result[:, u, t].
        ggml_tensor* gate_bias_t = nullptr;
        ggml_tensor* up_bias_t = nullptr;
        ggml_tensor* down_bias_t = nullptr;
        const std::int64_t gate_bias_dim = fused_gate_up
            ? static_cast<std::int64_t>(2) * n_ff
            : static_cast<std::int64_t>(n_ff);
        if (gate_bias != nullptr)
        {
            gate_bias_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, gate_bias_dim, num_experts);
        }
        if (!fused_gate_up && up_bias != nullptr)
        {
            up_bias_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_ff, num_experts);
        }
        if (down_bias != nullptr)
        {
            down_bias_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_dim, num_experts);
        }

        if (hidden_t == nullptr || hidden_out_t == nullptr ||
            gate_w == nullptr || down_w == nullptr ||
            ids_t == nullptr || weights_t == nullptr ||
            (!fused_gate_up && up_w == nullptr))
        {
            set_last_error("MoE prefill: failed to create ggml tensors.");
            return 0;
        }

        // --- Build graph ---
        // Reshape input to [hidden_dim, 1, seq_len] so mul_mat_id can broadcast
        // it across the n_used expert slots per token.
        ggml_tensor* cur_3d = ggml_reshape_3d(ctx, hidden_t, hidden_dim, 1, seq_len);
        if (cur_3d == nullptr)
        {
            set_last_error("MoE prefill: failed to reshape input.");
            return 0;
        }

        ggml_tensor* gate_proj = nullptr;
        ggml_tensor* up_proj = nullptr;

        if (fused_gate_up)
        {
            // gate_w shape: [hidden_dim, 2*n_ff, num_experts]
            // result shape: [2*n_ff, n_used, seq_len]
            ggml_tensor* gate_up = ggml_mul_mat_id(ctx, gate_w, cur_3d, ids_t);
            if (gate_up == nullptr)
            {
                set_last_error("MoE prefill: ggml_mul_mat_id(gate_up) failed.");
                return 0;
            }
            if (gate_bias_t != nullptr)
            {
                gate_up = ggml_add_id(ctx, gate_up, gate_bias_t, ids_t);
                if (gate_up == nullptr)
                {
                    set_last_error("MoE prefill: ggml_add_id(gate_up_bias) failed.");
                    return 0;
                }
            }
            // Split into two views over the leading dim.
            // gate = view at offset 0, up = view at offset n_ff * nb[0].
            gate_proj = ggml_view_3d(ctx, gate_up,
                static_cast<std::int64_t>(n_ff),
                gate_up->ne[1],
                gate_up->ne[2],
                gate_up->nb[1],
                gate_up->nb[2],
                0);
            up_proj = ggml_view_3d(ctx, gate_up,
                static_cast<std::int64_t>(n_ff),
                gate_up->ne[1],
                gate_up->ne[2],
                gate_up->nb[1],
                gate_up->nb[2],
                static_cast<std::size_t>(n_ff) * gate_up->nb[0]);
        }
        else
        {
            gate_proj = ggml_mul_mat_id(ctx, gate_w, cur_3d, ids_t);
            up_proj = ggml_mul_mat_id(ctx, up_w, cur_3d, ids_t);
            if (gate_proj == nullptr || up_proj == nullptr)
            {
                set_last_error("MoE prefill: failed to build gate/up projections.");
                return 0;
            }
            if (gate_bias_t != nullptr)
            {
                gate_proj = ggml_add_id(ctx, gate_proj, gate_bias_t, ids_t);
            }
            if (up_bias_t != nullptr)
            {
                up_proj = ggml_add_id(ctx, up_proj, up_bias_t, ids_t);
            }
        }

        if (gate_proj == nullptr || up_proj == nullptr)
        {
            set_last_error("MoE prefill: failed to build gate/up projections.");
            return 0;
        }

        ggml_tensor* activated = nullptr;
        switch (activation_type)
        {
            case MOE_ACT_SWIGLU_SPLIT:
                activated = ggml_swiglu_split(ctx, gate_proj, up_proj);
                break;
            case MOE_ACT_SWIGLU_OAI:
                activated = ggml_swiglu_oai(ctx, gate_proj, up_proj, oai_alpha, oai_limit);
                break;
            case MOE_ACT_GEGLU_SPLIT:
                activated = ggml_geglu_split(ctx, gate_proj, up_proj);
                break;
            default:
                set_last_error("MoE prefill: unknown activation_type.");
                return 0;
        }

        if (activated == nullptr)
        {
            set_last_error("MoE prefill: SwiGLU activation node creation failed.");
            return 0;
        }

        // Down projection: [hidden_dim, n_used, seq_len]
        ggml_tensor* down_proj = ggml_mul_mat_id(ctx, down_w, activated, ids_t);
        if (down_proj == nullptr)
        {
            set_last_error("MoE prefill: ggml_mul_mat_id(down) failed.");
            return 0;
        }

        if (down_bias_t != nullptr)
        {
            down_proj = ggml_add_id(ctx, down_proj, down_bias_t, ids_t);
            if (down_proj == nullptr)
            {
                set_last_error("MoE prefill: ggml_add_id(down_bias) failed.");
                return 0;
            }
        }

        // Apply per-(token,expert) routing weights via broadcast multiply.
        ggml_tensor* weighted = ggml_mul(ctx, down_proj, weights_t);
        if (weighted == nullptr)
        {
            set_last_error("MoE prefill: routing weight multiply failed.");
            return 0;
        }

        // Aggregate experts: sum across the n_used dim → [hidden_dim, seq_len].
        ggml_tensor* moe_out = build_expert_aggregation(ctx, weighted, hidden_dim, n_used, seq_len);
        if (moe_out == nullptr)
        {
            set_last_error("MoE prefill: expert aggregation failed.");
            return 0;
        }

        ggml_tensor* output = nullptr;
        if (residual_mode)
        {
            // Gemma 4 MoE post-aggregation chain:
            //   residual += rms_norm(moe_out, eps) * post_ffw_norm_2.weight
            //
            // Mirrors the Gemma4Model.cs MoEForward + RMSNormOp + Ops.Add
            // sequence (`moeOut → postMoeNormed → mlpOut += postMoeNormed`)
            // but as one fused graph dispatch instead of three. The RMSNorm
            // reduces along ne0 (hidden_dim per token) which is exactly the
            // unweighted RMSNorm Gemma 4 expects on the MoE output before
            // the post_ffw_norm_2 weighted scale.
            //
            // ggml_rms_norm operates along the leading dim (ne0); `moe_out`
            // is [hidden_dim, seq_len], so each column (token) is normalised
            // independently.  ggml_mul broadcasts the 1-D post_norm weight
            // across the seq_len dimension, identical semantics to ollama /
            // llama.cpp's RMSNorm-with-weight pattern (ggml_rms_norm + ggml_mul
            // is the canonical idiom because no fused op exists).
            ggml_tensor* moe_normed = ggml_rms_norm(ctx, moe_out, post_norm_eps);
            if (moe_normed == nullptr)
            {
                set_last_error("MoE prefill: post-norm RMSNorm node creation failed.");
                return 0;
            }
            moe_normed = ggml_mul(ctx, moe_normed, post_norm_w_t);
            if (moe_normed == nullptr)
            {
                set_last_error("MoE prefill: post-norm weight multiply failed.");
                return 0;
            }

            // hidden_out_t holds the residual (dense FFN output) value here:
            // it's bound below to `residual_in_out`. The graph reads it,
            // adds the normed MoE result, and writes the sum back over the
            // same backend buffer (in-place residual fold).
            ggml_tensor* added = ggml_add(ctx, hidden_out_t, moe_normed);
            if (added == nullptr)
            {
                set_last_error("MoE prefill: residual add failed.");
                return 0;
            }
            output = ggml_cpy(ctx, added, hidden_out_t);
        }
        else
        {
            output = ggml_cpy(ctx, moe_out, hidden_out_t);
        }
        if (output == nullptr)
        {
            set_last_error("MoE prefill: output copy node creation failed.");
            return 0;
        }
        ggml_set_output(output);

        ggml_cgraph* graph = ggml_new_graph_custom(ctx, 1024, false);
        if (graph == nullptr)
        {
            set_last_error("MoE prefill: failed to create graph.");
            return 0;
        }
        ggml_build_forward_expand(graph, output);

        // --- Bind tensors to backend memory ---
        // hidden_in / hidden_out (or residual_in_out): try zero-copy host-pointer
        // mapping first (saves ~hidden_bytes upload + download per call on
        // Metal). The cached host-ptr buffer path reuses the same Metal buffer
        // across calls when the C# Tensor's underlying storage pointer is
        // stable.
        std::vector<BufferHandle> ephem_buffers;
        std::vector<WeightUploadInfo> uploads;
        bool hidden_in_zero_copy = false;
        bool hidden_out_zero_copy = false;
        // The output buffer in residual mode is the residual_in_out pointer;
        // in standalone mode it's hidden_out. Same shape (seq_len*hidden_dim
        // F32) and same kernel write pattern, so the binding code below is
        // identical.
        float* const out_buffer_ptr = residual_mode ? residual_in_out : hidden_out;

        if (dev != nullptr && hidden_bytes >= 4096)
        {
            // cacheable=false: a cached host-ptr buffer cross-pollinates with
            // every other op that touches the same C# Tensor.Storage pointer
            // (because the cache is keyed on the host data pointer, not the
            // ggml_tensor identity), and the resulting Metal buffer reuse
            // degrades scheduler decisions in practice. Pay the small per-call
            // wrap cost; the freed buffers are released by BufferHandle's
            // destructor at the end of the call.
            ggml_backend_buffer_t buf_in = nullptr;
            if (try_get_host_ptr_buffer(g_backend, dev, hidden_in, hidden_bytes, false, buf_in))
            {
                ephem_buffers.emplace_back(buf_in);
                if (ggml_backend_tensor_alloc(buf_in, hidden_t, hidden_in) == GGML_STATUS_SUCCESS)
                {
                    hidden_in_zero_copy = true;
                }
            }

            ggml_backend_buffer_t buf_out = nullptr;
            if (try_get_host_ptr_buffer(g_backend, dev, out_buffer_ptr, hidden_bytes, false, buf_out))
            {
                ephem_buffers.emplace_back(buf_out);
                if (ggml_backend_tensor_alloc(buf_out, hidden_out_t, out_buffer_ptr) == GGML_STATUS_SUCCESS)
                {
                    hidden_out_zero_copy = true;
                }
            }
        }

        // Quantized expert weights: cacheable read-only path (reused across
        // calls), falling back to host-ptr mapping or deferred upload.
        bind_readonly_weight_3d(ctx, g_backend, dev, gate_data,
            static_cast<std::size_t>(gate_total_bytes), gate_w, uploads, ephem_buffers);
        if (!fused_gate_up)
        {
            bind_readonly_weight_3d(ctx, g_backend, dev, up_data,
                static_cast<std::size_t>(up_total_bytes), up_w, uploads, ephem_buffers);
        }
        bind_readonly_weight_3d(ctx, g_backend, dev, down_data,
            static_cast<std::size_t>(down_total_bytes), down_w, uploads, ephem_buffers);

        // Allocate any tensors not bound above (hidden in/out fallback, ids,
        // weights, biases). ggml_backend_alloc_ctx_tensors will skip
        // already-allocated tensors and allocate the rest into a single
        // backend buffer.
        BufferHandle backend_buffer(ggml_backend_alloc_ctx_tensors(ctx, g_backend));
        if (backend_buffer.value == nullptr)
        {
            set_last_error("MoE prefill: failed to allocate backend buffer for graph tensors.");
            return 0;
        }

        // Drain pending GPU work before the upcoming CPU memcpys so we don't
        // race with in-flight zero-copy GPU writes targeting `hidden_in` /
        // `selected_experts` / `routing_weights` from the previous op.
        host_read_barrier();

        for (const WeightUploadInfo& u : uploads)
        {
            if (u.needs_upload && u.tensor != nullptr && u.host_data != nullptr && u.bytes > 0)
            {
                ggml_backend_tensor_set(u.tensor, u.host_data, 0, u.bytes);
            }
        }

        if (!hidden_in_zero_copy)
        {
            ggml_backend_tensor_set(hidden_t, hidden_in, 0, hidden_bytes);
        }

        // In residual mode the kernel reads the residual value out of
        // hidden_out_t (so it can add the normed MoE result to it). When the
        // backend buffer is host-mapped we already see the caller's data; if
        // not, we have to push the residual to device first.
        if (residual_mode && !hidden_out_zero_copy)
        {
            ggml_backend_tensor_set(hidden_out_t, residual_in_out, 0, hidden_bytes);
        }

        // post_ffw_norm_2.weight is small (hidden_dim * 4B). Upload directly
        // each call - reusing the cacheable-tensor path for it would require
        // the C# layer to keep the pointer stable across the model's lifetime
        // and to use a separate ggml_tensor identity per layer to avoid
        // cross-layer aliasing in the buffer cache. Easier and just as cheap
        // to push the few KB over the bus on every call.
        if (post_norm_w_t != nullptr)
        {
            ggml_backend_tensor_set(post_norm_w_t, post_norm_w, 0,
                static_cast<std::size_t>(hidden_dim) * sizeof(float));
        }

        // ids and routing weights are always small; copy from C# host arrays.
        ggml_backend_tensor_set(ids_t, selected_experts, 0,
            static_cast<std::size_t>(seq_len) *
            static_cast<std::size_t>(n_used) * sizeof(std::int32_t));
        ggml_backend_tensor_set(weights_t, routing_weights, 0,
            static_cast<std::size_t>(seq_len) *
            static_cast<std::size_t>(n_used) * sizeof(float));

        // Per-expert biases: small (n_experts × ffn_dim or hidden_dim × 4 bytes).
        // Upload directly from the caller's host arrays each call.
        if (gate_bias_t != nullptr)
        {
            ggml_backend_tensor_set(gate_bias_t, gate_bias, 0,
                static_cast<std::size_t>(gate_bias_dim) *
                static_cast<std::size_t>(num_experts) * sizeof(float));
        }
        if (up_bias_t != nullptr)
        {
            ggml_backend_tensor_set(up_bias_t, up_bias, 0,
                static_cast<std::size_t>(n_ff) *
                static_cast<std::size_t>(num_experts) * sizeof(float));
        }
        if (down_bias_t != nullptr)
        {
            ggml_backend_tensor_set(down_bias_t, down_bias, 0,
                static_cast<std::size_t>(hidden_dim) *
                static_cast<std::size_t>(num_experts) * sizeof(float));
        }

        ggml_status status = ggml_backend_graph_compute(g_backend, graph);
        if (status != GGML_STATUS_SUCCESS)
        {
            set_last_error("MoE prefill: ggml_backend_graph_compute failed.");
            return 0;
        }

        // Result publish. With a zero-copy host-mapped output buffer, the
        // backend has already written into the host-visible memory and we
        // only need to mark pending GPU work for the next host barrier.
        // Otherwise queue an async blit download (Metal) or sync+copy back.
        // out_buffer_ptr is hidden_out (standalone) or residual_in_out
        // (residual mode); the kernel wrote into the same backend tensor
        // (hidden_out_t) in both cases so the finalize path is identical.
        if (hidden_out_zero_copy)
        {
            finalize_compute(true, hidden_out_t, out_buffer_ptr, hidden_bytes);
        }
        else
        {
            finalize_compute_with_download(hidden_out_t, out_buffer_ptr, hidden_bytes);
        }

        clear_last_error();
        return 1;
    }

    // Backwards-compatible thin wrapper: invokes the unified graph builder
    // in standalone mode (writes moe_out to a dedicated buffer, no residual
    // fold). Kept for the existing call sites in C# (Qwen 3.5 / GPT-OSS /
    // the original Gemma 4 MoE non-residual path) that haven't been
    // restructured to pass a residual tensor.
    int moe_ffn_prefill_swiglu_quant_f32_impl(
        float* hidden_in,
        float* hidden_out,
        int seq_len,
        int hidden_dim,
        int n_ff,
        int num_experts,
        int n_used,
        const std::int32_t* selected_experts,
        const float* routing_weights,
        void* gate_data, int gate_type, std::int64_t gate_ne0, std::int64_t gate_ne1, std::int64_t gate_total_bytes,
        void* up_data,   int up_type,   std::int64_t up_ne0,   std::int64_t up_ne1,   std::int64_t up_total_bytes,
        void* down_data, int down_type, std::int64_t down_ne0, std::int64_t down_ne1, std::int64_t down_total_bytes,
        const float* gate_bias,
        const float* up_bias,
        const float* down_bias,
        int activation_type,
        float oai_alpha,
        float oai_limit)
    {
        MoEFFNGraphParams p;
        p.hidden_in = hidden_in;
        p.hidden_out = hidden_out;
        p.seq_len = seq_len;
        p.hidden_dim = hidden_dim;
        p.n_ff = n_ff;
        p.num_experts = num_experts;
        p.n_used = n_used;
        p.selected_experts = selected_experts;
        p.routing_weights = routing_weights;
        p.gate_data = gate_data;
        p.gate_type = gate_type;
        p.gate_ne0 = gate_ne0;
        p.gate_ne1 = gate_ne1;
        p.gate_total_bytes = gate_total_bytes;
        p.up_data = up_data;
        p.up_type = up_type;
        p.up_ne0 = up_ne0;
        p.up_ne1 = up_ne1;
        p.up_total_bytes = up_total_bytes;
        p.down_data = down_data;
        p.down_type = down_type;
        p.down_ne0 = down_ne0;
        p.down_ne1 = down_ne1;
        p.down_total_bytes = down_total_bytes;
        p.gate_bias = gate_bias;
        p.up_bias = up_bias;
        p.down_bias = down_bias;
        p.activation_type = activation_type;
        p.oai_alpha = oai_alpha;
        p.oai_limit = oai_limit;
        return moe_ffn_prefill_graph_impl(p);
    }

    // Residual-fused variant: invokes the unified graph builder in residual
    // mode so the caller's residual buffer ends up containing
    // `residual + rms_norm(moe_out, eps) * post_norm_w` after one graph
    // dispatch. Designed for Gemma 4 MoE which routes every layer through
    // dense_FFN + post_norm_1 + (MoE_FFN -> post_norm_2 -> add). The kernel
    // collapses the post_norm_2 + add into the same dispatch as the MoE FFN.
    int gemma4_moe_geglu_residual_f32_impl(
        float* hidden_in,
        float* residual_in_out,
        const float* post_norm_w,
        float post_norm_eps,
        int seq_len,
        int hidden_dim,
        int n_ff,
        int num_experts,
        int n_used,
        const std::int32_t* selected_experts,
        const float* routing_weights,
        void* gate_data, int gate_type, std::int64_t gate_ne0, std::int64_t gate_ne1, std::int64_t gate_total_bytes,
        void* up_data,   int up_type,   std::int64_t up_ne0,   std::int64_t up_ne1,   std::int64_t up_total_bytes,
        void* down_data, int down_type, std::int64_t down_ne0, std::int64_t down_ne1, std::int64_t down_total_bytes,
        const float* gate_bias,
        const float* up_bias,
        const float* down_bias,
        int activation_type,
        float oai_alpha,
        float oai_limit)
    {
        MoEFFNGraphParams p;
        p.hidden_in = hidden_in;
        p.residual_in_out = residual_in_out;
        p.post_norm_w = post_norm_w;
        p.post_norm_eps = post_norm_eps;
        p.seq_len = seq_len;
        p.hidden_dim = hidden_dim;
        p.n_ff = n_ff;
        p.num_experts = num_experts;
        p.n_used = n_used;
        p.selected_experts = selected_experts;
        p.routing_weights = routing_weights;
        p.gate_data = gate_data;
        p.gate_type = gate_type;
        p.gate_ne0 = gate_ne0;
        p.gate_ne1 = gate_ne1;
        p.gate_total_bytes = gate_total_bytes;
        p.up_data = up_data;
        p.up_type = up_type;
        p.up_ne0 = up_ne0;
        p.up_ne1 = up_ne1;
        p.up_total_bytes = up_total_bytes;
        p.down_data = down_data;
        p.down_type = down_type;
        p.down_ne0 = down_ne0;
        p.down_ne1 = down_ne1;
        p.down_total_bytes = down_total_bytes;
        p.gate_bias = gate_bias;
        p.up_bias = up_bias;
        p.down_bias = down_bias;
        p.activation_type = activation_type;
        p.oai_alpha = oai_alpha;
        p.oai_limit = oai_limit;
        return moe_ffn_prefill_graph_impl(p);
    }
} // namespace

extern "C"
{
    // Fused MoE FFN prefill (Mixture-of-Experts SwiGLU FFN).
    //
    // Computes the standard MoE FFN body in a single GGML graph dispatch:
    //   for each token t and used expert u with weight w_{t,u}:
    //     gate = W_gate[expert_id(t,u)] @ hidden[t]
    //     up   = W_up  [expert_id(t,u)] @ hidden[t]   (or split from fused gate_up)
    //     act  = swiglu(gate, up)                     (split or oai variant)
    //     down = W_down[expert_id(t,u)] @ act
    //   moe_out[t] = sum_u w_{t,u} * down_{t,u}
    //
    // hidden_in / hidden_out: contiguous f32 of size seq_len * hidden_dim. The
    //                         caller owns both buffers; the kernel writes only
    //                         `hidden_out` (it does NOT add to a residual).
    // selected_experts:        i32 [seq_len, n_used] in row-major layout
    //                          (selected_experts[t * n_used + u] is the expert
    //                          index used by token t for slot u).
    // routing_weights:         f32 [seq_len, n_used] in the same layout.
    // gate / up / down weights: stacked per-expert quantized blocks with the
    //                          on-disk GGUF layout for `_exps.weight` tensors,
    //                          i.e. [ne0, ne1, num_experts] contiguous. For
    //                          mmap'd models the C# layer passes the base
    //                          pointer of expert 0 for free; otherwise the
    //                          caller stacks them at model load time.
    // up_data == nullptr:      indicates a fused gate_up weight (gate_ne1 must
    //                          be 2 * n_ff). Mirrors llama.cpp's gate_up_exps
    //                          path where supported.
    // activation_type:         0 = silu(gate) * up (regular SwiGLU split)
    //                          1 = swiglu_oai (gpt-oss clamped variant). Uses
    //                              oai_alpha and oai_limit.
    //                          2 = gelu(gate) * up (Gemma 4 MoE GEGLU split).
    TSG_EXPORT int TSGgml_MoEFFNPrefillSwiGLUQuantF32(
        float* hidden_in,
        float* hidden_out,
        int seq_len,
        int hidden_dim,
        int n_ff,
        int num_experts,
        int n_used,
        const std::int32_t* selected_experts,
        const float* routing_weights,
        void* gate_data, int gate_type, std::int64_t gate_ne0, std::int64_t gate_ne1, std::int64_t gate_total_bytes,
        void* up_data,   int up_type,   std::int64_t up_ne0,   std::int64_t up_ne1,   std::int64_t up_total_bytes,
        void* down_data, int down_type, std::int64_t down_ne0, std::int64_t down_ne1, std::int64_t down_total_bytes,
        const float* gate_bias,
        const float* up_bias,
        const float* down_bias,
        int activation_type,
        float oai_alpha,
        float oai_limit)
    {
        try
        {
            return moe_ffn_prefill_swiglu_quant_f32_impl(
                hidden_in, hidden_out, seq_len, hidden_dim, n_ff,
                num_experts, n_used, selected_experts, routing_weights,
                gate_data, gate_type, gate_ne0, gate_ne1, gate_total_bytes,
                up_data,   up_type,   up_ne0,   up_ne1,   up_total_bytes,
                down_data, down_type, down_ne0, down_ne1, down_total_bytes,
                gate_bias, up_bias, down_bias,
                activation_type, oai_alpha, oai_limit);
        }
        catch (const std::exception& ex)
        {
            set_last_error(ex.what());
            return 0;
        }
        catch (...)
        {
            set_last_error("Unknown error in MoE FFN prefill.");
            return 0;
        }
    }

    // Gemma 4 MoE residual-fused GEGLU kernel.
    //
    // Computes the standard MoE FFN body PLUS the post_ffw_norm_2 RMSNorm
    // (with weighted scale post_norm_w) PLUS the residual add into the
    // dense-FFN-output buffer in a SINGLE GGML graph dispatch:
    //
    //   moe_out      = moe_ffn(hidden_in, gate, up, down, ids, weights)   [batched mul_mat_id]
    //   moe_normed   = rms_norm(moe_out, eps) * post_norm_w
    //   residual_in_out += moe_normed                                     [in-place residual]
    //
    // Saves two device dispatches per MoE layer (the standalone RMSNorm and
    // the standalone Add) versus the previous TryMoEFusedGEGLU + RMSNorm +
    // Add sequence in C#. Mirrors the analogous Qwen 3.5 single-token kernel
    // TSGgml_MoEExpertsSwiGLUResidual which folds the routed-expert + shared-
    // expert + residual into one graph submission.
    //
    // hidden_in (in):           [seq_len, hidden_dim] F32 contiguous - the MoE
    //                           input (pre_ffw_norm_2 output of the residual
    //                           stream in Gemma 4 MoE layers).
    // residual_in_out (in/out): [seq_len, hidden_dim] F32 contiguous - the
    //                           dense FFN output that the kernel adds the
    //                           normed MoE result to. The kernel writes the
    //                           sum back over the same buffer.
    // post_norm_w (in):         [hidden_dim] F32 - the post_ffw_norm_2.weight
    //                           tensor.
    // post_norm_eps:            RMSNorm epsilon (ModelConfig.Eps in C#).
    // selected_experts:         i32 [seq_len, n_used] in row-major layout.
    // routing_weights:          f32 [seq_len, n_used] in row-major layout.
    //                           Per-expert post-down scales (Gemma 4's
    //                           ffn_down_exps.scale) must be folded into
    //                           these by the caller before the call.
    // gate / up / down weights: per-expert stacked quantized blocks, identical
    //                           layout to TSGgml_MoEFFNPrefillSwiGLUQuantF32.
    //                           up_data == nullptr signals a fused gate_up
    //                           weight (gate_ne1 = 2 * n_ff) - the kernel
    //                           splits internally.
    // activation_type:          MoE GLU activation. For Gemma 4 MoE this is
    //                           always 2 (GEGLU split = gelu(gate) * up); the
    //                           parameter is passed through for symmetry with
    //                           the standalone kernel and to leave room for
    //                           future variants without an ABI change.
    TSG_EXPORT int TSGgml_Gemma4MoEGEGLUResidualF32(
        float* hidden_in,
        float* residual_in_out,
        const float* post_norm_w,
        float post_norm_eps,
        int seq_len,
        int hidden_dim,
        int n_ff,
        int num_experts,
        int n_used,
        const std::int32_t* selected_experts,
        const float* routing_weights,
        void* gate_data, int gate_type, std::int64_t gate_ne0, std::int64_t gate_ne1, std::int64_t gate_total_bytes,
        void* up_data,   int up_type,   std::int64_t up_ne0,   std::int64_t up_ne1,   std::int64_t up_total_bytes,
        void* down_data, int down_type, std::int64_t down_ne0, std::int64_t down_ne1, std::int64_t down_total_bytes,
        const float* gate_bias,
        const float* up_bias,
        const float* down_bias,
        int activation_type,
        float oai_alpha,
        float oai_limit)
    {
        try
        {
            return gemma4_moe_geglu_residual_f32_impl(
                hidden_in, residual_in_out, post_norm_w, post_norm_eps,
                seq_len, hidden_dim, n_ff,
                num_experts, n_used, selected_experts, routing_weights,
                gate_data, gate_type, gate_ne0, gate_ne1, gate_total_bytes,
                up_data,   up_type,   up_ne0,   up_ne1,   up_total_bytes,
                down_data, down_type, down_ne0, down_ne1, down_total_bytes,
                gate_bias, up_bias, down_bias,
                activation_type, oai_alpha, oai_limit);
        }
        catch (const std::exception& ex)
        {
            set_last_error(ex.what());
            return 0;
        }
        catch (...)
        {
            set_last_error("Unknown error in Gemma 4 MoE GEGLU residual kernel.");
            return 0;
        }
    }
}
