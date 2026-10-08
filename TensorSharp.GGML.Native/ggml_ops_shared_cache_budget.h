// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>

namespace tsg
{
    // Callbacks only perform accounting. They must not throw or call native
    // cache APIs. The registry lock serializes callbacks and attach/detach;
    // cache locks may precede it, but this registry never takes a cache lock.
    class SharedCacheCharge
    {
    public:
        using Reserve = std::uint64_t (*)(void*, int, int, std::int64_t);
        using Commit = int (*)(void*, std::uint64_t);
        using Release = void (*)(void*, std::uint64_t);
    private:
        struct Registry {
            std::mutex mutex;
            void* context = nullptr;
            Reserve reserve = nullptr;
            Commit commit = nullptr;
            Release release = nullptr;
            std::uint64_t live = 0;
        };
        static Registry& registry()
        {
            // Native cache teardown can run from C runtime finalizers. Keep the
            // bookkeeping gate alive for as long as any cache entry can exist.
            static Registry* value = new Registry;
            return *value;
        }
        std::uint64_t token_ = 0;
        std::size_t reserved_ = 0;
        bool live_ = false;
        bool committed_ = false;
    public:
        SharedCacheCharge() = default;
        SharedCacheCharge(const SharedCacheCharge&) = delete;
        SharedCacheCharge& operator=(const SharedCacheCharge&) = delete;

        static bool attach(void* context, Reserve reserve, Commit commit, Release release)
        {
            if (context == nullptr || reserve == nullptr || commit == nullptr || release == nullptr) return false;
            auto& r = registry();
            std::lock_guard<std::mutex> lock(r.mutex);
            if (r.reserve != nullptr || r.live != 0) return false;
            r.context = context; r.reserve = reserve; r.commit = commit; r.release = release;
            return true;
        }
        static bool detach(void* context)
        {
            auto& r = registry();
            std::lock_guard<std::mutex> lock(r.mutex);
            if (r.context != context || r.reserve == nullptr || r.live != 0) return false;
            r.context = nullptr; r.reserve = nullptr; r.commit = nullptr; r.release = nullptr;
            return true;
        }
        static std::shared_ptr<SharedCacheCharge> reserve(int rank, int kind, std::size_t bytes)
        {
            if (bytes == 0 || bytes > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) return {};
            auto charge = std::make_shared<SharedCacheCharge>();
            auto& r = registry();
            std::lock_guard<std::mutex> lock(r.mutex);
            if (r.reserve != nullptr) {
                charge->token_ = r.reserve(r.context, rank, kind, static_cast<std::int64_t>(bytes));
                if (charge->token_ == 0) return {};
            }
            // Also count unconfigured allocations: an attach must never miss a
            // buffer that is being allocated, or adopt already-live cache bytes.
            ++r.live;
            charge->live_ = true;
            charge->reserved_ = bytes;
            return charge;
        }
        bool commit(std::size_t actual)
        {
            if (!live_ || (token_ != 0 && actual > reserved_)) return false;
            if (committed_) return true;
            auto& r = registry();
            std::lock_guard<std::mutex> lock(r.mutex);
            if (token_ != 0 && !r.commit(r.context, token_)) return false;
            committed_ = true;
            return true;
        }
        ~SharedCacheCharge()
        {
            if (!live_) return;
            auto& r = registry();
            std::lock_guard<std::mutex> lock(r.mutex);
            if (token_ != 0) r.release(r.context, token_);
            --r.live;
        }
    };
}
