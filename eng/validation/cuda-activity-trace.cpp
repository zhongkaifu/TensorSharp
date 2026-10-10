// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
// Diagnostic-only CUPTI activity recorder. Build against the installed CUDA SDK.
#include <cupti.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <mutex>

namespace {
std::mutex output_mutex;
FILE* output = nullptr;
std::atomic<unsigned long long> errors{0}, dropped{0}, records{0};
constexpr CUpti_ActivityKind kinds[] = {CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,
    CUPTI_ACTIVITY_KIND_RUNTIME, CUPTI_ACTIVITY_KIND_MEMCPY};
bool active = false;
void quoted(const char* text) {
    std::fputc('"', output);
    for (const unsigned char* p = (const unsigned char*)(text ? text : ""); *p; ++p) {
        if (*p == '"' || *p == '\\') std::fputc('\\', output);
        if (*p < 32) std::fprintf(output, "\\u%04x", *p);
        else std::fputc(*p, output);
    }
    std::fputc('"', output);
}
bool check(CUptiResult result) {
    if (result == CUPTI_SUCCESS) return true;
    ++errors; const char* message = nullptr; cuptiGetResultString(result, &message);
    std::fprintf(stderr, "[cuda-activity] %s\n", message ? message : "CUPTI error"); return false;
}
void CUPTIAPI requested(uint8_t** buffer, size_t* size, size_t* max_records) {
    *size = 8 * 1024 * 1024; *max_records = 0;
    *buffer = (uint8_t*)std::malloc(*size);
    if (!*buffer) { ++errors; *size = 0; }
}
void CUPTIAPI completed(CUcontext context, uint32_t stream, uint8_t* buffer, size_t, size_t valid) {
    std::lock_guard<std::mutex> lock(output_mutex);
    if (!output) { ++errors; std::free(buffer); return; }
    CUpti_Activity* record = nullptr;
    for (;;) {
        auto status = cuptiActivityGetNextRecord(buffer, valid, &record);
        if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) break;
        if (!check(status)) break;
        ++records;
        if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
            auto* k = (CUpti_ActivityKernel9*)record;
            std::fprintf(output, "{\"kind\":\"kernel\",\"start\":%llu,\"end\":%llu,\"device\":%u,\"stream\":%u,\"graph\":%u,\"name\":",
                (unsigned long long)k->start, (unsigned long long)k->end, k->deviceId, k->streamId, k->graphId);
            quoted(k->name); std::fprintf(output, "}\n");
        } else if (record->kind == CUPTI_ACTIVITY_KIND_RUNTIME) {
            auto* api = (CUpti_ActivityAPI*)record; const char* name = nullptr;
            check(cuptiGetCallbackName(CUPTI_CB_DOMAIN_RUNTIME_API, api->cbid, &name));
            std::fprintf(output, "{\"kind\":\"api\",\"start\":%llu,\"end\":%llu,\"name\":",
                (unsigned long long)api->start, (unsigned long long)api->end);
            quoted(name); std::fprintf(output, "}\n");
        } else if (record->kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
            auto* copy = (CUpti_ActivityMemcpy6*)record;
            std::fprintf(output, "{\"kind\":\"copy\",\"start\":%llu,\"end\":%llu,\"device\":%u,\"bytes\":%llu,\"copy_kind\":%u}\n",
                (unsigned long long)copy->start, (unsigned long long)copy->end, copy->deviceId,
                (unsigned long long)copy->bytes, copy->copyKind);
        }
    }
    size_t lost = 0; check(cuptiActivityGetNumDroppedRecords(context, stream, &lost)); dropped += lost;
    if (std::ferror(output)) ++errors;
    std::free(buffer);
}
}

extern "C" int TsCudaTraceStart(const char* path) {
    if (active || output || !path) return 0;
    output = std::fopen(path, "wx"); if (!output) return 0;
    active = true;
    if (!check(cuptiActivityRegisterCallbacks(requested, completed))) return 0;
    for (auto kind : kinds) if (!check(cuptiActivityEnable(kind))) return 0;
    return 1;
}
extern "C" int TsCudaTraceMark(const char* label) {
    uint64_t now = 0; if (!active || !check(cuptiGetTimestamp(&now))) return 0;
    std::lock_guard<std::mutex> lock(output_mutex);
    std::fprintf(output, "{\"kind\":\"mark\",\"timestamp\":%llu,\"label\":", (unsigned long long)now);
    quoted(label); std::fprintf(output, "}\n"); return !std::ferror(output);
}
extern "C" int TsCudaTraceStop() {
    if (!active) return 0;
    for (auto kind : kinds) check(cuptiActivityDisable(kind));
    check(cuptiActivityFlushAll(0));
    std::lock_guard<std::mutex> lock(output_mutex);
    std::fprintf(output, "{\"kind\":\"summary\",\"records\":%llu,\"dropped\":%llu,\"errors\":%llu}\n",
        records.load(), dropped.load(), errors.load());
    if (std::fclose(output)) ++errors;
    output = nullptr; active = false;
    return errors == 0 && dropped == 0 && records > 0;
}
