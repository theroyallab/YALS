#ifndef SHARED_RESOURCE_BUNDLE_HPP
#define SHARED_RESOURCE_BUNDLE_HPP

#include <atomic>
#include "readback_buffer.hpp"
#include "samplers.hpp"
#include <iostream>

/*
 * An atomic reference counted shared resource bundle for cooperative resources.
 */

struct SharedResourceBundle {
    ReadbackBuffer* readback_buffer{nullptr};
    llama_sampler* sampler{nullptr};

    std::atomic<unsigned> ref_count{1};
};

// C API
// Free with resource_bundle_release -- this is a shared ptr.
SharedResourceBundle* resource_bundle_make() {
    const auto bundle = new SharedResourceBundle{};
    bundle->readback_buffer = new ReadbackBuffer{};
    bundle->sampler = sampler_make();
    return bundle;
}

SharedResourceBundle* resource_bundle_ref_acquire(SharedResourceBundle* bundle) {
    bundle->ref_count.fetch_add(1, std::memory_order_relaxed);
    return bundle;
}

// C API
void resource_bundle_release(SharedResourceBundle* bundle) {
    if (!bundle) return;

    if ((bundle->ref_count.fetch_sub(1, std::memory_order_acq_rel)) == 1) {
        delete bundle->readback_buffer;
        sampler_free(bundle->sampler);
        delete bundle;
    }
}

#endif //SHARED_RESOURCE_BUNDLE_HPP
