// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "../glm_metadata_owner.h"
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <new>

static void require(bool pass, const char * message) {
    if (!pass) { std::fprintf(stderr, "FAIL: %s\n", message); std::exit(1); }
}

// Portable ownership test: opaque GGUF/tensor contexts are represented by real
// malloc payloads. Instrumented frees validate order and exact-once cleanup;
// they do not emulate or qualify upstream GGUF parsing/allocation internals.
static int live_file = 0, live_tensors = 0, freed_file = 0, freed_tensors = 0;
static void free_file(gguf_context * context) {
    require(live_file == 1 && live_tensors == 1, "file released before tensor metadata");
    --live_file; ++freed_file; std::free(context);
}
static void free_tensors(ggml_context * context) {
    require(live_tensors == 1 && live_file == 0, "tensor metadata released after file");
    --live_tensors; ++freed_tensors; std::free(context);
}
using owner = tsg_glm::metadata_owner<free_file, free_tensors>;

static void initialize(owner & value, bool with_file = true) {
    value.tensors = reinterpret_cast<ggml_context *>(std::malloc(4096));
    require(value.tensors != nullptr, "allocate real tensor metadata payload");
    ++live_tensors;
    if (with_file) {
        value.file = reinterpret_cast<gguf_context *>(std::malloc(8192));
        require(value.file != nullptr, "allocate real file metadata payload");
        ++live_file;
    }
}

template<class T> struct fail_allocator : std::allocator<T> {
    template<class U> struct rebind { using other = fail_allocator<U>; };
    fail_allocator() = default;
    template<class U> fail_allocator(const fail_allocator<U> &) {}
    T * allocate(size_t) { throw std::bad_alloc(); }
};

int main() {
    for (int retry = 0; retry < 64; ++retry) {
        bool caught = false;
        try {
            owner metadata;
            initialize(metadata);
            std::map<int, int, std::less<int>, fail_allocator<std::pair<const int, int>>> sources;
            sources.emplace(17, 31); // Fail at the actual source-map allocation boundary.
        } catch (const std::bad_alloc &) { caught = true; }
        require(caught && live_file == 0 && live_tensors == 0,
                "source-map allocation failure unwinds both physical metadata owners");
        {
            owner metadata;
            initialize(metadata);
            std::map<int, int> sources;
            sources.emplace(17, 31);
            require(sources.at(17) == 31, "source map retry succeeds");
            metadata.reset();
            metadata.reset();
        }
        require(live_file == 0 && live_tensors == 0 && freed_file == (retry + 1) * 2 && freed_tensors == freed_file,
                "normal/reset/destruction releases every payload exactly once");
    }
    try {
        owner partial;
        initialize(partial, false);
        throw std::bad_alloc();
    } catch (const std::bad_alloc &) {}
    require(live_file == 0 && live_tensors == 0 && freed_tensors == freed_file + 1,
            "partially populated metadata owner also unwinds");
    std::puts("PASS GLM metadata ownership: source-map allocation failure, real payload release order, partial initialization, retry");
    return 0;
}
