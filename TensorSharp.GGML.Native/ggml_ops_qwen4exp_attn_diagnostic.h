// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace tsg {
struct Qwen4ExpAttnDiagnosticConfig {
    enum class Scope { both, solo, arena };
    int layer = -1, position = -1;
    std::string directory;
    Scope scope = Scope::both;
    bool enabled() const noexcept { return layer >= 0; }
    bool matches(bool arena, int candidate_layer) const noexcept {
        return enabled() && layer == candidate_layer
            && (scope == Scope::both || scope == (arena ? Scope::arena : Scope::solo));
    }
};

inline int qwen4exp_diagnostic_integer(const char* text, int maximum) {
    if (!text || !*text) throw std::invalid_argument("Qwen4Exp attention diagnostic: missing integer");
    for (const char* p = text; *p; ++p)
        if (*p < '0' || *p > '9') throw std::invalid_argument("Qwen4Exp attention diagnostic: invalid integer");
    errno = 0; char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (errno == ERANGE || *end || value < 0 || value > maximum)
        throw std::invalid_argument("Qwen4Exp attention diagnostic: integer out of range");
    return static_cast<int>(value);
}

inline Qwen4ExpAttnDiagnosticConfig qwen4exp_attn_diagnostic_parse(
    const char* layer, const char* position, const char* directory, const char* scope = nullptr) {
    Qwen4ExpAttnDiagnosticConfig result;
    if (!layer && !position && !directory && !scope) return result;
    result.layer = qwen4exp_diagnostic_integer(layer, 4095);
    result.position = qwen4exp_diagnostic_integer(position, INT_MAX);
    if (!directory || !*directory) throw std::invalid_argument("Qwen4Exp attention diagnostic: output directory required");
    result.directory = directory;
    if (scope) {
        if (std::strcmp(scope, "solo") == 0) result.scope = Qwen4ExpAttnDiagnosticConfig::Scope::solo;
        else if (std::strcmp(scope, "arena") == 0) result.scope = Qwen4ExpAttnDiagnosticConfig::Scope::arena;
        else if (std::strcmp(scope, "both") != 0)
            throw std::invalid_argument("Qwen4Exp attention diagnostic: scope must be both, solo or arena");
    }
    return result;
}

inline const Qwen4ExpAttnDiagnosticConfig& qwen4exp_attn_diagnostic_config() {
    static const auto value = qwen4exp_attn_diagnostic_parse(
        std::getenv("TS_Q4E_ATTN_TRACE_LAYER"), std::getenv("TS_Q4E_ATTN_TRACE_POSITION"),
        std::getenv("TS_Q4E_ATTN_TRACE_DIR"), std::getenv("TS_Q4E_ATTN_TRACE_SCOPE"));
    return value;
}

inline std::size_t qwen4exp_attn_diagnostic_bytes(const int64_t (&ne)[4], std::size_t element) {
    constexpr std::size_t maximum = 64 * 1024 * 1024;
    if (element == 0 || element > maximum)
        throw std::runtime_error("Qwen4Exp attention diagnostic: invalid element size");
    std::size_t bytes = element;
    for (const auto dimension : ne) {
        if (dimension <= 0 || static_cast<unsigned long long>(dimension) > maximum / bytes)
            throw std::runtime_error("Qwen4Exp attention diagnostic: invalid or excessive tensor extent");
        bytes *= static_cast<std::size_t>(dimension);
    }
    return bytes;
}

// Normal mode records only tensors whose storage is already protected by the
// production graph or persistent KV owner. It adds no graph nodes or flags.
// Other intermediates contribute layout metadata only: reading their reused
// storage after execution would not be a valid observation of their values.
struct Qwen4ExpAttnDiagnostic {
    struct Item { const char* name; ggml_tensor* tensor; bool readable; };
    int layer = -1, slot = 0;
    std::vector<Item> items;
    void add(const char* name, ggml_tensor* tensor, bool already_live) {
        const auto& config = qwen4exp_attn_diagnostic_config();
        if (!config.enabled() || !tensor) return;
        items.push_back({name, tensor, already_live});
    }

    void dump(const char* path, int position, int caller) const {
        const auto& config = qwen4exp_attn_diagnostic_config();
        if (!config.matches(std::strcmp(path, "arena") == 0, layer) || position != config.position) return;
        static std::atomic<unsigned long long> sequence{0};
        const auto invocation = sequence.fetch_add(1, std::memory_order_relaxed);
        for (const auto& item : items) {
            const auto* t = item.tensor;
            std::fprintf(stderr, "[q4e-attn-layout] path=%s call=%llu layer=%d pos=%d caller=%d slot=%d tensor=%s readable=%d op=%s flags=%d data=%p src0=%p src1=%p nb=%zu,%zu,%zu,%zu\n",
                path, invocation, layer, position, caller, slot, item.name, item.readable ? 1 : 0,
                ggml_op_name(t->op), t->flags, t->data, t->src[0] ? t->src[0]->data : nullptr,
                t->src[1] ? t->src[1]->data : nullptr, t->nb[0], t->nb[1], t->nb[2], t->nb[3]);
            if (!item.readable) continue;
            if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16)
                throw std::runtime_error("Qwen4Exp attention diagnostic: unsupported tensor type");
            const auto element = ggml_type_size(t->type);
            int64_t ne[4] = {t->ne[0], t->ne[1], t->ne[2], t->ne[3]};
            if (std::strcmp(item.name, "k_cache") == 0 || std::strcmp(item.name, "v_cache") == 0)
                ne[1] = std::min<int64_t>(ne[1], static_cast<int64_t>(position) + 1);
            if (t->nb[0] != element)
                throw std::runtime_error("Qwen4Exp attention diagnostic: non-scalar column stride");
            std::vector<unsigned char> bytes(qwen4exp_attn_diagnostic_bytes(ne, element));
            std::size_t output = 0;
            const auto row_bytes = static_cast<std::size_t>(ne[0]) * element;
            for (int64_t i3 = 0; i3 < ne[3]; ++i3) for (int64_t i2 = 0; i2 < ne[2]; ++i2) {
                const auto offset = static_cast<std::size_t>(i3) * t->nb[3] + static_cast<std::size_t>(i2) * t->nb[2];
                if (t->nb[1] == row_bytes) {
                    const auto size = row_bytes * static_cast<std::size_t>(ne[1]);
                    ggml_backend_tensor_get(t, bytes.data() + output, offset, size); output += size;
                } else for (int64_t i1 = 0; i1 < ne[1]; ++i1) {
                    ggml_backend_tensor_get(t, bytes.data() + output, offset + static_cast<std::size_t>(i1) * t->nb[1], row_bytes);
                    output += row_bytes;
                }
            }
            unsigned long long hash = 1469598103934665603ULL;
            for (const auto value : bytes) hash = (hash ^ value) * 1099511628211ULL;
            const auto file = config.directory + "/q4e-attn-" + path + "-" + std::to_string(invocation)
                + "-l" + std::to_string(layer) + "-p" + std::to_string(position)
                + "-r" + std::to_string(caller) + "-" + item.name + ".bin";
            std::unique_ptr<FILE, decltype(&std::fclose)> out(ggml_fopen(file.c_str(), "wb"), std::fclose);
            if (!out || std::fwrite(bytes.data(), 1, bytes.size(), out.get()) != bytes.size() || std::fflush(out.get()) != 0)
                throw std::runtime_error("Qwen4Exp attention diagnostic: write failed");
            std::fprintf(stderr, "[q4e-attn-diagnostic] path=%s call=%llu layer=%d pos=%d caller=%d slot=%d tensor=%s type=%s ne=%lld,%lld,%lld,%lld bytes=%zu hash=%016llx file=%s\n",
                path, invocation, layer, position, caller, slot, item.name, ggml_type_name(t->type),
                static_cast<long long>(ne[0]), static_cast<long long>(ne[1]), static_cast<long long>(ne[2]), static_cast<long long>(ne[3]), bytes.size(), hash, file.c_str());
        }
    }
};
}
