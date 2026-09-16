// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once
#include "ggml.h"
#include "ggml-backend.h"
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void require(bool value, const char * message)
{
    if (!value) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}

// Logical values are kept separately from padded device storage so the oracle
// cannot reproduce a bad kernel's stride calculation. Padding contains NaNs.
struct input_tensor
{
    ggml_tensor * storage;
    ggml_tensor * tensor;
    size_t offset;
    std::vector<unsigned char> bytes;
    std::vector<float> values;

    input_tensor(ggml_context * ctx, ggml_type type, std::array<int64_t, 4> ne,
                 bool padded = false, bool interleaved = false)
    {
        const size_t element = ggml_type_size(type);
        const size_t nb0 = element * (interleaved ? 2 : 1);
        const size_t nb1 = nb0 * ne[0] + (padded ? 3 * element : 0);
        const size_t nb2 = nb1 * (ne[1] + (padded ? 2 : 0));
        const size_t nb3 = nb2 * (ne[2] + (padded ? 1 : 0));
        offset = padded ? 2 * element : 0;
        bytes.resize(offset + nb3 * ne[3], 0xff);
        storage = ggml_new_tensor_1d(ctx, type, bytes.size() / element);
        tensor = ggml_view_4d(ctx, storage, ne[0], ne[1], ne[2], ne[3], nb1, nb2, nb3, offset);
        tensor->nb[0] = nb0;
        values.resize(ne[0] * ne[1] * ne[2] * ne[3]);
    }

    size_t logical(int64_t x, int64_t y, int64_t z, int64_t w) const
    {
        return x + tensor->ne[0] * (y + tensor->ne[1] * (z + tensor->ne[2] * w));
    }

    float at(int64_t x, int64_t y, int64_t z = 0, int64_t w = 0) const
    {
        return values[logical(x, y, z, w)];
    }

    void upload()
    {
        for (int64_t w = 0; w < tensor->ne[3]; ++w)
        for (int64_t z = 0; z < tensor->ne[2]; ++z)
        for (int64_t y = 0; y < tensor->ne[1]; ++y)
        for (int64_t x = 0; x < tensor->ne[0]; ++x)
        {
            float & value = values[logical(x, y, z, w)];
            unsigned char * dst = bytes.data() + offset + x * tensor->nb[0] + y * tensor->nb[1]
                + z * tensor->nb[2] + w * tensor->nb[3];
            if (tensor->type == GGML_TYPE_F32) std::memcpy(dst, &value, sizeof(value));
            else if (tensor->type == GGML_TYPE_F16)
            {
                const auto encoded = ggml_fp32_to_fp16(value);
                value = ggml_fp16_to_fp32(encoded);
                std::memcpy(dst, &encoded, sizeof(encoded));
            }
            else if (tensor->type == GGML_TYPE_BF16)
            {
                const auto encoded = ggml_fp32_to_bf16(value);
                value = ggml_bf16_to_fp32(encoded);
                std::memcpy(dst, &encoded, sizeof(encoded));
            }
            else
            {
                require(tensor->type == GGML_TYPE_I32, "Unsupported test input type");
                const int32_t encoded = static_cast<int32_t>(value);
                std::memcpy(dst, &encoded, sizeof(encoded));
            }
        }
        ggml_backend_tensor_set(storage, bytes.data(), 0, bytes.size());
    }

    void check_unchanged() const
    {
        std::vector<unsigned char> actual(bytes.size());
        ggml_backend_tensor_get(storage, actual.data(), 0, actual.size());
        require(actual == bytes, "Precision matmul modified an input or its padding");
    }
};

