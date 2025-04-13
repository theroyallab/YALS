#ifndef GENERATION_RESOURCES_HPP
#define GENERATION_RESOURCES_HPP

#include <atomic>
#include "readback_buffer.hpp"
#include "samplers.hpp"

/*
 * An atomic reference counted shared resource bundle for cooperative resources.
 */

struct GenerationResources {
    ReadbackBuffer* readback_buffer{nullptr};
    llama_sampler* sampler{nullptr};

    std::atomic<unsigned> ref_count{1};
};

// C API
// Free with resource_bundle_release -- this is a shared ptr.
GenerationResources* generation_resources_make() {
    const auto bundle = new GenerationResources{};
    bundle->readback_buffer = new ReadbackBuffer{};
    bundle->sampler = sampler_make();
    return bundle;
}

GenerationResources* generation_resources_ref_acquire(GenerationResources* resources) {
    resources->ref_count.fetch_add(1, std::memory_order_relaxed);
    return resources;
}

// C API
void generation_resources_release(GenerationResources* resources) {
    if (!resources) {
        return;
    }

    if ((resources->ref_count.fetch_sub(1, std::memory_order_acq_rel)) == 1) {
        delete resources->readback_buffer;
        sampler_free(resources->sampler);
        delete resources;
    }
}

#endif //GENERATION_RESOURCES_HPP
