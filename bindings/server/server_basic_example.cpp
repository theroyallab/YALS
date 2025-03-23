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
    sampler = sampler_temp(sampler, 2);
    sampler = sampler_dist(sampler, 1337);

    const auto processor = processor_make(model, ctx, 4);

    const auto readback_buffer = readback_create_buffer();

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

    auto lark_grammar = R"(
// Define the start rule
start: json_string

// The exact JSON string with fixed format
json_string: "{\n    \"action\" : [\"" ACTION_CONTENT "\"],\n    \"mood\" : \"" EMOTION "\",\n    \"magazine capacity\" : \"" CAPACITY_CONTENT "\"\n}"

// Content restrictions
ACTION_CONTENT: /[a-zA-Z0-9 ,]{1,15}/
CAPACITY_CONTENT: /[0-9]+( rounds| bullets| shots)?/
EMOTION: "happy" | "sad" | "angry" | "excited" | "bored" | "anxious" | "calm" | "confused"
       | "curious" | "depressed" | "ecstatic" | "fearful" | "grateful" | "hopeful"
       | "irritated" | "jealous" | "peaceful" | "proud" | "surprised" | "tired"
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
        nullptr,
        0,
        nullptr,
        0,
        nullptr,
        0,
        lark_grammar);

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
