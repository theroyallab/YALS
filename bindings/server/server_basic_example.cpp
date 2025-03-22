#include <iostream>
#include "common.h"
#include "c_library.h"
#include "llama.h"

int main() {
    const auto idk = new float(0.0);
    const auto model = model_load(
        "/home/blackroot/Desktop/YALS/YALS/models/PocketDoc_Dans-PersonalityEngine-V1.2.0-24b-Q6_K_L.gguf",
        999,
        idk,
        nullptr
        );

    const auto ctx = ctx_make(model, 1024, 999, 512, false, -1, false, 0, 0, 0.0f);
    if (!model || !ctx) {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }

    std::cout << "Model and context loaded successfully" << std::endl;

    auto sampler = sampler_make();
    sampler = sampler_greedy(sampler);

    const auto processor = processor_make(model, ctx, 4);

    const auto readback_buffer = readback_create_buffer();

    const auto prompt = R"(<|im_start|>system
Respond with *actions* *words* *thoughts*
<|im_end|>
<|im_start|>user
Hi how are you?
<|im_end|>
<|im_start|>assistant
)";

    const char* seq[] = {"*"};

    processor_submit_work(
        processor,
        prompt,
        sampler,
        readback_buffer,
        100,
        0,
        1337,
        seq,
        1,
        nullptr,
        0,
        nullptr,
        0);

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

    return 0;
}
