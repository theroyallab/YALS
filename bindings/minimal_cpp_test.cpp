#include "server/c_library.h"
#include "llama.h"
#include <chrono>
#include <iostream>
#include <vector>
#include <thread>

std::pair<llama_model*, llama_context*> initialize_model(const std::string& model_path, const int ctx_size = 2048, const int batch_size = 512, const int threads = 4) {
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 999;
    llama_model* model = llama_model_load_from_file(model_path.c_str(), model_params);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = ctx_size;
    ctx_params.n_batch = batch_size;
    ctx_params.n_threads = threads;

    llama_context* ctx = llama_init_from_model(model, ctx_params);
    return {model, ctx};
}

struct ReadbackBuffer {
    unsigned last_readback_index {0};
    bool buffer_finished_write {false};
    char* status_buffer = nullptr;
    std::vector<char*>* data = new std::vector<char*>();
    std::vector<llama_token>* ids = new std::vector<llama_token>();
};

inline std::string readback_debug_check_buffer(const ReadbackBuffer* buffer)
{
    if (!buffer || !buffer->data) {
        return "";
    }

    std::string result;

    // Concatenate all text in the buffer
    for (const auto & i : *buffer->data) {
        result += i;
    }

    return result;
}

void run_example() {
    const auto idk = new float(0.0);
    const auto model = model_load(
        "/home/blackroot/Desktop/tab/yals-internal/Llama-3.2-1B-Instruct-IQ4_XS.gguf",
        999,
        idk,
        nullptr
        );

    const auto ctx = ctx_make(model, 1024, 999, 512, false, -1, false, 0, 0, 0.0f);

    if (!model || !ctx) {
        return;
    }

    const auto processor = processor_make(model, ctx, 4);

    llama_sampler* sampler = sampler_make();
    sampler = sampler_greedy(sampler);

    const char* seq[] = {"python"};

    llama_sampler* sampler2 = sampler_make();
    sampler2 = sampler_temp(sampler2, 0.0);
    sampler2 = sampler_dist(sampler2, 1337);

    ReadbackBuffer* readback_job1 = readback_create_buffer();
    ReadbackBuffer* readback_job2 = readback_create_buffer();

    //processor_cancel_work(processor, 0);

    const int job_1 = processor_submit_work(processor,
        "Output six unique emojis",
        sampler2,
        readback_job1,
        500,
        250,
        1337,
        nullptr,
        0,
        nullptr,
        0,
        nullptr,
        0);

    const int job_2 = processor_submit_work(processor,
        "What's your name?",
        sampler,
        readback_job2,
        100,
        0,
        1337,
        nullptr,
        0,
        nullptr,
        0,
        nullptr,
        0);

    // std::this_thread::sleep_for(std::chrono::milliseconds(500));
    // processor.cancel_work(job_2);
    // std::cout << "Cancelled j1" << std::endl;
    //
    // std::this_thread::sleep_for(std::chrono::milliseconds(500));
    // processor.cancel_work(job_1);
    // std::cout << "Cancelled j2" << std::endl;

    std::cout << "Running generation for 5 seconds..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(5));

    const auto out1 = readback_debug_check_buffer(readback_job1);
    const auto out2= readback_debug_check_buffer(readback_job2);
    std::cout << "************************\n\n\n" << out1 << std::endl;
    std::cout << "************************\n\n\n" << out2 << std::endl;


    llama_sampler_free(sampler);
    llama_sampler_free(sampler2);
    llama_free(ctx);
    llama_model_free(model);
}

int main() {
    llama_backend_init();
    run_example();
    llama_backend_free();
    return 0;
}