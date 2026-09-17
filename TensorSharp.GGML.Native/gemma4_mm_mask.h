// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once

#include <algorithm>
#include <cstddef>

// Attention-mask rows for Gemma 4's fused whole-model prefill kernels
// (TSGgml_Gemma4ModelVerify, TSGgml_Gemma4MoEModelVerify), including the
// bidirectional attention of multimodal soft tokens (image, video frame, audio).
//
// A prefill call covers the chunk of N tokens at logical positions
// [start_pos, start_pos + N). `is_except`, when non-null, has one byte per
// CHUNK token: 1 marks a soft token. The rule every row implements is:
//
//   query q may read key k  iff  (k <= q, and k is inside q's sliding window
//                                 when the layer has one)
//                            or  (q and k are both soft tokens OF THIS CHUNK)
//
// The second clause is what makes an image span attend to itself in both
// directions. Soft keys from an EARLIER call are already in the cache and reached
// by the first clause only: they were computed without seeing this chunk, exactly
// as a cold prefill that cut its chunk at the same place computed them.
//
// Every buffer a kernel attends holds the chunk's own fresh keys as its LAST N
// real keys, whatever precedes them: nothing (start_pos 0), the global linear
// cache [0, start_pos), an unwrapped sliding-window cache [0, start_pos), or the
// previous window gathered from a wrapped ring [start_pos - prev, start_pos).
// So a key's chunk index is its buffer index minus the number of real keys before
// the chunk, and that is the only thing the soft-token clause needs. Before
// multimodal chunks were accepted at start_pos > 0 the kernels compared the
// buffer index itself, which is the chunk index only when nothing precedes it.
//
// At start_pos 0 both functions compute exactly what the kernels computed
// before (tests/gemma4_mm_mask_test.cpp keeps a verbatim copy of the old rows).
namespace tsg_gemma4_mask
{
    // Whether chunk token `chunk_index` is a soft token of the chunk.
    inline bool chunk_soft(const unsigned char* is_except, int n, int chunk_index)
    {
        return is_except != nullptr && chunk_index >= 0 && chunk_index < n && is_except[chunk_index] != 0;
    }

    // One row of a buffer-relative mask: `kv_len` key slots of which the first
    // `valid_len` are real (the rest is flash-attention padding), the last `n` real
    // ones being the chunk. `qi` is the query's chunk index (rows qi >= n are batch
    // padding and never soft). `window` > 0 applies a sliding-window low bound to
    // the causal clause; 0 means the buffer is already the window (or the layer is
    // global).
    template <typename T>
    inline void fill_relative_row(T* row, int kv_len, int qi, int n, int valid_len, int window,
                                  const unsigned char* is_except, T zero, T neg_inf)
    {
        const int n_past = valid_len - n;
        const int threshold = n_past + qi;
        const int low = (window > 0) ? (threshold - window + 1) : 0;
        if (!chunk_soft(is_except, n, qi))
        {
            // The visible keys are one contiguous band [lo, hi]; fill it directly
            // (a per-element loop over [0, kv_len) blocks the GPU at long prefill).
            const int lo = (low > 0) ? low : 0;
            const int hi = std::min(threshold, valid_len - 1);
            std::fill(row, row + kv_len, neg_inf);
            if (hi >= lo && lo < kv_len)
                std::fill(row + lo, row + std::min(hi + 1, kv_len), zero);
            return;
        }
        for (int ki = 0; ki < kv_len; ki++)
        {
            const bool causal = (ki < valid_len) && (ki <= threshold) && !(window > 0 && ki < low);
            const bool bidi = chunk_soft(is_except, n, ki - n_past);
            row[ki] = (causal || bidi) ? zero : neg_inf;
        }
    }

    // One row of an absolute-position mask (the MoE kernel's tile mask): the query
    // sits at logical position `q_abs`, key slot ki at logical `k_start + ki`, and
    // the chunk starts at `start_pos`. `window` <= 0 means no sliding-window bound.
    template <typename T>
    inline void fill_absolute_row(T* row, int k_len, int q_abs, int k_start, int window,
                                  int start_pos, int n, const unsigned char* is_except, T zero, T neg_inf)
    {
        if (!chunk_soft(is_except, n, q_abs - start_pos))
        {
            const int lo = (window > 0) ? std::max(0, q_abs - window + 1 - k_start) : 0;
            int hi = q_abs - k_start;
            if (hi > k_len - 1) hi = k_len - 1;
            std::fill(row, row + k_len, neg_inf);
            if (hi >= lo && lo < k_len)
                std::fill(row + lo, row + hi + 1, zero);
            return;
        }
        for (int ki = 0; ki < k_len; ki++)
        {
            const int k_abs = k_start + ki;
            const bool causal = (k_abs <= q_abs) && !(window > 0 && k_abs < q_abs - window + 1);
            const bool bidi = chunk_soft(is_except, n, k_abs - start_pos);
            row[ki] = (causal || bidi) ? zero : neg_inf;
        }
    }
}
