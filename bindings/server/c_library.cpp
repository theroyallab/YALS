#include "c_library.h"
#include "processor.hpp"
#include <sstream>

// Implementation of processor interface functions
int processor_submit_work(
    Processor* processor,
    const char* prompt,
    llama_sampler* sampler,
    ReadbackBuffer* readback_buffer,
    const int max_tokens,
    const int min_tokens,
    const unsigned seed,
    const char** rewind_strings,
    const unsigned num_rewind_strings,
    const char** stopping_strings,
    const unsigned num_stopping_strings,
    const int32_t* stopping_tokens,
    const unsigned num_stopping_tokens) {

    const std::string prompt_as_string(prompt);

    const InferenceArgs args(
        sampler,
        max_tokens,
        min_tokens,
        seed,
        rewind_strings,
        num_rewind_strings,
        stopping_strings,
        num_stopping_strings,
        stopping_tokens,
        num_stopping_tokens
    );

    return processor->submit_work(
        prompt_as_string,
        args,
        readback_buffer);
}

bool processor_cancel_work(Processor* processor, const int request_id_to_cancel) {
    return processor->cancel_work(request_id_to_cancel);
}

Processor* processor_make(llama_model* model, llama_context* ctx, const int num_processor_slots) {
    return new Processor(model, ctx, num_processor_slots);
}

llama_model* model_load(
    const char* model_path,
    const int32_t number_gpu_layers,
    const float* tensor_split,
    const llama_progress_callback callback)
{
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = number_gpu_layers;
    model_params.progress_callback = callback;

    model_params.split_mode = LLAMA_SPLIT_MODE_LAYER;
    model_params.tensor_split = tensor_split;

    llama_model* model = llama_model_load_from_file(model_path, model_params);

    return model;
}

float model_get_freq_base(const llama_model* model) {
    static auto freqBaseKey = "general.rope_freq_base";

    const int32_t bufSize = llama_model_meta_val_str(model, freqBaseKey, nullptr, 0) + 1;
    if (bufSize <= 1) {
        return 10000.0f;
    }

    std::vector<char> buffer(bufSize);
    const int32_t written = llama_model_meta_val_str(model, freqBaseKey, buffer.data(), bufSize);
    if (written <= 0) {
        return 10000.0f;
    }

    try {
        std::stringstream ss(buffer.data());
        ss.imbue(std::locale::classic());
        float value;
        ss >> value;

        if (ss.fail()) {
            return 10000.0f;
        }

        return value;
    } catch (...) {
        return 10000.0f;
    }
}

void model_free(llama_model* model)
{
    llama_model_free(model);
}

llama_token model_vocab_bos(const llama_model* model)
{
    return llama_vocab_bos(&model->vocab);
}

llama_token model_vocab_eos(const llama_model* model)
{
    return llama_vocab_eos(&model->vocab);
}

llama_token model_vocab_eot(const llama_model* model)
{
    return llama_vocab_eot(&model->vocab);
}

const char* model_vocab_token_to_string(const llama_model* model, const llama_token token) {
    return llama_vocab_get_text(&model->vocab, token);
}

llama_context* ctx_make(
    llama_model* model,
    const unsigned context_length,
    const int32_t number_gpu_layers,
    const unsigned num_batches,
    const bool flash_attn,
    const float rope_freq_base,
    const bool use_yarn,
    int k_cache_quant_type,
    int v_cache_quant_type,
    const float kv_defrag_threshold
) {
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = context_length;
    ctx_params.n_batch = num_batches;
    ctx_params.n_ubatch = num_batches;
    ctx_params.no_perf = false;
    ctx_params.flash_attn = flash_attn;

    ctx_params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_NONE;

    const float freqBaseTrain = model_get_freq_base(model);

    // Yarn, allegedly ext_factor -1 to default to model cfg, but it looks sussy.
    // Only set linear RoPE if freq base is greater than the trained base
    if (use_yarn) {
        ctx_params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_YARN;
        ctx_params.yarn_ext_factor = -1;
    } else if (rope_freq_base > freqBaseTrain) {
        ctx_params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_LINEAR;
        ctx_params.rope_freq_base = rope_freq_base;
        ctx_params.rope_freq_scale = 0;
    }

    // Decrease CPU threads if model is fully offloaded on GPU
    if (number_gpu_layers >= llama_model_n_layer(model) || number_gpu_layers == -1) {
        ctx_params.n_threads = 1;
        ctx_params.n_threads_batch = 1;
    }

    ctx_params.type_k = static_cast<ggml_type>(k_cache_quant_type);
    ctx_params.type_v = static_cast<ggml_type>(v_cache_quant_type);
    ctx_params.defrag_thold = kv_defrag_threshold;
    llama_context* ctx = llama_init_from_model(model, ctx_params);

    return ctx;
}

uint32_t ctx_max_seq_len(const llama_context* ctx)
{
    return llama_n_ctx(ctx);
}

void ctx_free(llama_context* ctx)
{
    llama_free(ctx);
}

void ctx_clear_kv(llama_context* ctx)
{
    llama_kv_self_clear(ctx);
}

int32_t* endpoint_tokenize(
    const llama_model* model,
    const char* prompt,
    const bool add_special,
    const bool parse_special) {

    const auto promptLength = static_cast<int32_t>(strlen(prompt));
    const int n_prompt = -llama_tokenize(&model->vocab, prompt, promptLength,
                                   nullptr, 0, add_special, parse_special);
    const auto tokenArray = new int32_t[n_prompt + 1];
    tokenArray[0] = n_prompt;

    if (llama_tokenize(&model->vocab, prompt, promptLength,
    tokenArray + 1, n_prompt + 1,
        add_special, parse_special) < 0) {
        return nullptr;
        }

    return tokenArray;
}

char* model_chat_template(const llama_model* model) {
    static auto tokenizerTemplateKey = "tokenizer.chat_template";
    const int32_t bufSize = llama_model_meta_val_str(model, tokenizerTemplateKey, nullptr, 0) + 1;

    // Return null if template doesn't exist
    if (bufSize <= 1) {
        return nullptr;
    }

    const auto buffer = new char[bufSize];
    llama_model_meta_val_str(model, tokenizerTemplateKey, buffer, bufSize);

    // Additional check to see if the buffer has data
    if (buffer[0] == '\0') {
        delete[] buffer;
        return nullptr;
    }

    return buffer;
}

char* endpoint_detokenize(
        const llama_model* model,
        const int32_t* tokens,
        const int32_t num_tokens,
        const int32_t max_text_size,
        const bool add_special,
        const bool parse_special) {
    const auto outText = new char[max_text_size];
    llama_detokenize(&model->vocab, tokens, num_tokens, outText, max_text_size, add_special, parse_special);
    return outText;
}

void endpoint_free_string(const char* str) {
    delete[] str;
}

void endpoint_free_tokens(const int32_t* tokens) {
    delete[] tokens;
}