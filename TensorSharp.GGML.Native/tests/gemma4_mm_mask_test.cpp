// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// Gemma 4's fused-prefill attention masks with multimodal soft tokens
// (gemma4_mm_mask.h), checked two ways:
//
//   1. At start_pos 0, and for text at any start_pos, every row is identical to
//      the rows the kernels built before media chunks were accepted after a
//      reused prefix. The old row builders are copied here verbatim (with the
//      start_pos == 0 gate the kernels applied to is_except).
//
//   2. A media chunk prefilled at start_pos P > 0 sees exactly what the same
//      queries see in a cold prefill of the whole prompt from position 0, in
//      every buffer layout the kernels attend at P: the global linear cache, an
//      unwrapped sliding-window cache, and the previous window gathered from a
//      wrapped ring prepended to the chunk (dense relative rows and MoE absolute
//      rows). The cold prefill is itself built with the old start_pos-0 rows, so
//      the reference is what the kernels already did, not a restatement of the
//      new arithmetic. For the gathered window the test also proves nothing the
//      cold row reads was left out of the buffer.
//
// The same check with the old kernels' start_pos gate is asserted to FAIL, so the
// property really distinguishes the fix.
#include "gemma4_mm_mask.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace
{

using Row = std::vector<int>;   // 1 = visible, 0 = masked
constexpr int ZERO = 1;
constexpr int NEG = 0;

int g_checks = 0;

void require(bool value, const char* message)
{
    ++g_checks;
    if (!value) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
}

// ---- verbatim copies of the pre-change row builders -------------------------

// ggml_ops_gemma4_verify.cpp get_causal_mask (also the MoE get_causal_mask),
// with `is_except = (start_pos == 0) ? is_except_arr : nullptr`.
void legacy_relative_row(int* row, int kvLen, int qi, int N, int validLen, int window,
                         const unsigned char* is_except_arr, int start_pos)
{
    const unsigned char* is_except = (start_pos == 0) ? is_except_arr : nullptr;
    const int nPast = validLen - N;
    const int threshold = nPast + qi;
    const int low = (window > 0) ? (threshold - window + 1) : 0;
    const bool q_except = is_except != nullptr && qi < N && is_except[qi] != 0;
    if (!q_except)
    {
        const int lo = (low > 0) ? low : 0;
        const int hi = std::min(threshold, validLen - 1);
        std::fill(row, row + kvLen, NEG);
        if (hi >= lo && lo < kvLen)
            std::fill(row + lo, row + std::min(hi + 1, kvLen), ZERO);
        return;
    }
    for (int ki = 0; ki < kvLen; ki++)
    {
        bool causal = (ki < validLen) && (ki <= threshold) && !(window > 0 && ki < low);
        bool bidi = q_except && ki < N && is_except[ki] != 0;
        row[ki] = (causal || bidi) ? ZERO : NEG;
    }
}

// ggml_ops_gemma4_moe.cpp get_tile_mask, same start_pos gate.
void legacy_absolute_row(int* row, int kLen, int gQ, int kStart, int window, int start_pos, int N,
                         const unsigned char* mm_is_except)
{
    const unsigned char* is_except = (start_pos == 0) ? mm_is_except : nullptr;
    if (is_except != nullptr && gQ < N && is_except[gQ] != 0)
    {
        for (int ki = 0; ki < kLen; ki++)
        {
            const int kAbs = kStart + ki;
            bool causal = (kAbs <= gQ) && !(window > 0 && kAbs < gQ - window + 1);
            bool bidi = kAbs < N && is_except[kAbs] != 0;
            row[ki] = (causal || bidi) ? ZERO : NEG;
        }
        return;
    }
    const int lo = (window > 0) ? std::max(0, gQ - window + 1 - kStart) : 0;
    int hi = gQ - kStart; if (hi > kLen - 1) hi = kLen - 1;
    std::fill(row, row + kLen, NEG);
    if (hi >= lo && lo < kLen) std::fill(row + lo, row + hi + 1, ZERO);
}

Row new_relative(int kvLen, int qi, int N, int validLen, int window, const unsigned char* ex)
{
    Row r((size_t) kvLen);
    tsg_gemma4_mask::fill_relative_row(r.data(), kvLen, qi, N, validLen, window, ex, ZERO, NEG);
    return r;
}

Row old_relative(int kvLen, int qi, int N, int validLen, int window, const unsigned char* ex, int start_pos)
{
    Row r((size_t) kvLen);
    legacy_relative_row(r.data(), kvLen, qi, N, validLen, window, ex, start_pos);
    return r;
}

Row new_absolute(int kLen, int q, int kStart, int window, int start_pos, int N, const unsigned char* ex)
{
    Row r((size_t) kLen);
    tsg_gemma4_mask::fill_absolute_row(r.data(), kLen, q, kStart, window, start_pos, N, ex, ZERO, NEG);
    return r;
}

Row old_absolute(int kLen, int q, int kStart, int window, int start_pos, int N, const unsigned char* ex)
{
    Row r((size_t) kLen);
    legacy_absolute_row(r.data(), kLen, q, kStart, window, start_pos, N, ex);
    return r;
}

// Random soft-token spans over [0, n): 0..3 spans, including spans touching
// either end of the chunk.
std::vector<unsigned char> random_spans(std::mt19937& rng, int n)
{
    std::vector<unsigned char> ex((size_t) n, 0);
    int spans = (int) (rng() % 4);
    for (int s = 0; s < spans; s++)
    {
        int len = 1 + (int) (rng() % std::max(1, n / 2));
        int at = (int) (rng() % (unsigned) n);
        if (rng() % 5 == 0) at = 0;
        if (rng() % 5 == 0) at = std::max(0, n - len);
        for (int i = at; i < std::min(n, at + len); i++) ex[(size_t) i] = 1;
    }
    return ex;
}

// ---- 1. identical rows where nothing was supposed to change ------------------

void check_unchanged_rows()
{
    std::mt19937 rng(20260917);
    for (int iter = 0; iter < 4000; iter++)
    {
        const int N = 1 + (int) (rng() % 96);
        const int NQ = N + (int) (rng() % 64);          // Vulkan batch padding rows
        const int window = (rng() % 3 == 0) ? 0 : 1 + (int) (rng() % 128);
        auto ex = random_spans(rng, N);

        // start_pos 0 media prefill: every buffer is the chunk (+ padding).
        {
            const int validLen = N;
            const int kvLen = validLen + (int) (rng() % 64);
            for (int qi = 0; qi < NQ; qi++)
                require(new_relative(kvLen, qi, N, validLen, window, ex.data())
                        == old_relative(kvLen, qi, N, validLen, window, ex.data(), 0),
                        "start_pos 0 relative row changed");
            for (int q = 0; q < N; q++)
            {
                const int kStart = (int) (rng() % (unsigned) (q + 1));
                const int kLen = 1 + (int) (rng() % (unsigned) (N - kStart));
                require(new_absolute(kLen, q, kStart, window, 0, N, ex.data())
                        == old_absolute(kLen, q, kStart, window, 0, N, ex.data()),
                        "start_pos 0 absolute row changed");
            }
        }

        // Text (no soft tokens) at any start_pos: unchanged too.
        {
            const int start_pos = (int) (rng() % 2000);
            const int prev = (rng() % 2) ? start_pos : std::min(start_pos, 1 + (int) (rng() % 512));
            const int validLen = prev + N;
            const int kvLen = validLen + (int) (rng() % 64);
            for (int qi = 0; qi < NQ; qi++)
                require(new_relative(kvLen, qi, N, validLen, window, nullptr)
                        == old_relative(kvLen, qi, N, validLen, window, nullptr, start_pos),
                        "text relative row changed");
            const int q = start_pos + (int) (rng() % (unsigned) N);
            const int kStart = start_pos - prev;
            const int kLen = prev + N;
            require(new_absolute(kLen, q, kStart, window, start_pos, N, nullptr)
                    == old_absolute(kLen, q, kStart, window, start_pos, N, nullptr),
                    "text absolute row changed");
        }
    }
}

// ---- 2. a chunk at P sees what a cold prefill sees ---------------------------

struct Mismatch { bool any = false; };

// Cold prefill of [0, T) with soft tokens `ex_full` (length T), built with the old
// start_pos-0 rows: whether logical query q may read logical key k on a layer with
// sliding window `window` (0 = global).
bool cold_visible_relative(int T, int window, const std::vector<unsigned char>& ex_full, int q, int k)
{
    // Global / unwrapped: the kernel reads the cache [0, T) with window 0 (the cache
    // view is the window); a wrapped local layer reads the fresh chunk with window W.
    const int w = (window > 0 && T > window) ? window : 0;
    Row r = old_relative(T, q, T, T, w, ex_full.data(), 0);
    if (k < 0 || k >= T) return false;
    // An unwrapped local layer (T <= W) still has every key inside the window.
    return r[(size_t) k] == ZERO;
}

bool cold_visible_absolute(int T, int window, const std::vector<unsigned char>& ex_full, int q, int k)
{
    const int w = (window > 0 && T > window) ? window : 0;
    Row r = old_absolute(T, q, 0, w, 0, T, ex_full.data());
    if (k < 0 || k >= T) return false;
    return r[(size_t) k] == ZERO;
}

// Compare the reuse rows of one layout against the cold prefill. `use_new` false
// runs the pre-change builders (with their start_pos gate) and reports whether a
// mismatch was found instead of failing.
bool reuse_matches_cold(int P, int N, int window, bool moe, const std::vector<unsigned char>& ex_chunk,
                        bool use_new)
{
    const int T = P + N;
    std::vector<unsigned char> ex_full((size_t) T, 0);
    std::copy(ex_chunk.begin(), ex_chunk.end(), ex_full.begin() + P);

    // The buffer the kernel attends for the chunk at P, as the logical position of
    // its first key, its real length, and the window its mask applies.
    int key_base, valid_len, mask_window;
    if (window == 0)                       { key_base = 0; valid_len = T; mask_window = 0; }        // global cache
    else if (T <= window)                  { key_base = 0; valid_len = T; mask_window = 0; }        // unwrapped SWA cache
    else                                                                                            // swaPrev
    {
        const int prev = std::min(window, P);
        key_base = P - prev; valid_len = prev + N; mask_window = window;
    }
    const int kv_len = valid_len + 7;      // flash-attention padding slots

    for (int qi = 0; qi < N; qi++)
    {
        const int q = P + qi;
        Row row;
        if (!moe || mask_window == 0)
            row = use_new ? new_relative(kv_len, qi, N, valid_len, mask_window, ex_chunk.data())
                          : old_relative(kv_len, qi, N, valid_len, mask_window, ex_chunk.data(), P);
        else
            row = use_new ? new_absolute(valid_len, q, key_base, mask_window, P, N, ex_chunk.data())
                          : old_absolute(valid_len, q, key_base, mask_window, P, N, ex_chunk.data());

        for (int ki = 0; ki < (int) row.size(); ki++)
        {
            const int k = key_base + ki;
            const bool want = ki < valid_len
                && (moe ? cold_visible_absolute(T, window, ex_full, q, k)
                        : cold_visible_relative(T, window, ex_full, q, k));
            if ((row[(size_t) ki] == ZERO) != want)
            {
                if (use_new)
                {
                    std::fprintf(stderr, "P=%d N=%d W=%d moe=%d q=%d k=%d got=%d want=%d\n",
                                 P, N, window, moe ? 1 : 0, q, k, row[(size_t) ki], want ? 1 : 0);
                    require(false, "a chunk at start_pos > 0 does not see what a cold prefill sees");
                }
                return false;
            }
        }
        // Nothing a cold query reads may be missing from the buffer.
        for (int k = 0; k < key_base; k++)
        {
            const bool cold = moe ? cold_visible_absolute(T, window, ex_full, q, k)
                                  : cold_visible_relative(T, window, ex_full, q, k);
            if (use_new)
                require(!cold, "the gathered window dropped a key the cold prefill reads");
            else if (cold)
                return false;
        }
    }
    return true;
}

void check_reuse_equals_cold()
{
    std::mt19937 rng(42);
    int old_failures = 0, media_cases = 0;
    for (int iter = 0; iter < 600; iter++)
    {
        const int window = (iter % 4 == 0) ? 0 : 16 + (int) (rng() % 48);
        // Prefixes below, at and past the window, so the ring is unwrapped,
        // exactly full, and wrapped.
        int P;
        switch (iter % 3)
        {
            case 0: P = 1 + (int) (rng() % 24); break;
            case 1: P = window > 0 ? window : 40; break;
            default: P = (window > 0 ? window : 40) + 1 + (int) (rng() % 200); break;
        }
        // The media spans fit in the window, as a Gemma 4 image (<= 280 soft
        // tokens) fits in its 512-token window.
        const int N = 2 + (int) (rng() % 48);
        std::vector<unsigned char> ex((size_t) N, 0);
        const int span_len = 1 + (int) (rng() % (unsigned) std::min(N, window > 0 ? window : N));
        const int at = (int) (rng() % (unsigned) (N - span_len + 1));
        for (int i = at; i < at + span_len; i++) ex[(size_t) i] = 1;

        for (bool moe : { false, true })
        {
            reuse_matches_cold(P, N, window, moe, ex, /*use_new=*/true);
            ++media_cases;
            if (!reuse_matches_cold(P, N, window, moe, ex, /*use_new=*/false))
                ++old_failures;
        }

        // Text chunks at P were already exact before; they must stay so.
        std::vector<unsigned char> none((size_t) N, 0);
        reuse_matches_cold(P, N, window, false, none, true);
        require(reuse_matches_cold(P, N, window, false, none, false), "text chunk at P was not exact before either");
    }
    // Every media case with a soft query that has a soft key ahead of it failed on
    // the old gate (it dropped the soft-token clause at start_pos > 0).
    std::printf("media chunks at start_pos > 0: %d cases, %d inexact with the old start_pos gate\n",
                media_cases, old_failures);
    require(old_failures > media_cases / 2, "the old start_pos gate should fail most media cases");
}

}  // namespace

int main()
{
    check_unchanged_rows();
    check_reuse_equals_cold();
    std::printf("gemma4 multimodal mask: %d checks passed\n", g_checks);
    return 0;
}
