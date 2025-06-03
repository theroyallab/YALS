#include <iostream>
#include "c_library.h"
#include "llama.h"
#include "generation_resources.hpp"

int main() {
    const auto idk = new float(0.0);
    const auto model = model_load(
        "/home/blackroot/Desktop/YALS/YALS/models/PocketDoc_Dans-PersonalityEngine-V1.2.0-24b-Q6_K_L.gguf",
        999,
        2,
        idk,
        nullptr,
        nullptr,
        true,
        true
        );

    const auto ctx = ctx_make(model, 1024, 999, 10, 512, false, -1, false, 0, 0, 0.0f);
    if (!model || !ctx) {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }

    std::cout << "Model and context loaded successfully" << std::endl;

    GenerationResources* gen_resources = generation_resources_make();
    auto readback_buffer = gen_resources->readback_buffer;

    auto sampler = gen_resources->sampler;
    sampler_temp(sampler, .5);
    sampler_dist(sampler, 1337);

    std::cout << "Porc s" << std::endl;

    Processor *processor = processor_make(model, ctx, 1);

    std::cout << "Porc up" << std::endl;

    const auto prompt = R"(<|im_start|>system
Respond with *actions* *words* *thoughts* in a json format, with
{
    "action" : ["first, second]",
    "mood" : "current mood from 20 mood choices",
    "magazine capacity" : "a number"
}
<|im_end|>
<|im_start|>user
Hi how are you?
<|im_end|>
<|im_start|>assistant
)";

    std::cout << "Inference" << std::endl;
    processor_submit_work(
        processor,
        prompt,
        gen_resources,
        100,
        0,
        1024,
        1337,
        nullptr,
        0,
        nullptr,
        0,
        nullptr,
        0,
        true);

    std::cout << "Starting model:" << std::endl;
    while (!readback_is_buffer_finished(readback_buffer)) {
        char* char_out;
        llama_token token;
        if (readback_read_next(readback_buffer, &char_out, &token)) {
            std::cout << char_out;
            std::cout.flush();
        }
    }

    const char* status = readback_read_status(readback_buffer);
    std::cout << status << std::endl;

    generation_resources_release(gen_resources);

    return 0;
}
