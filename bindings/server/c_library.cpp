#include "c_library.h"

#include <map>

#include "processor.hpp"
#include <sstream>

#include "log.h"

// Implementation of processor interface functions
int processor_submit_work(
    Processor* processor,
    const char* prompt,
    GenerationResources* gen_resources,
    const int max_tokens,
    const int min_tokens,
    const uint32_t max_slot_n_ctx,
    const unsigned seed,
    const char** rewind_strings,
    const unsigned num_rewind_strings,
    const char** stopping_strings,
    const unsigned num_stopping_strings,
    const int32_t* stopping_tokens,
    const unsigned num_stopping_tokens,
    const bool add_special) {

    const std::string prompt_as_string(prompt);
    const InferenceArgs args(
        gen_resources,
        max_tokens,
        min_tokens,
        max_slot_n_ctx,
        seed,
        rewind_strings,
        num_rewind_strings,
        stopping_strings,
        num_stopping_strings,
        stopping_tokens,
        num_stopping_tokens,
        add_special
    );

    return processor->submit_work(
        prompt_as_string,
        args);
}

bool processor_cancel_work(Processor* processor, const int request_id_to_cancel) {
    return processor->cancel_work(request_id_to_cancel);
}

Processor* processor_make(llama_model* model, llama_context* ctx, llama_memory_t mem, const int num_processor_slots) {
    return new Processor(model, ctx, mem, num_processor_slots);
}

void processor_free(const Processor* processor) {
    delete processor;
}

// Simplified version from common args.cpp
std::vector<llama_model_tensor_buft_override> tensor_type_split(const std::string& value, std::vector<char*>& leaked_strings) {
    std::vector<llama_model_tensor_buft_override> tensor_buft_overrides;

    std::map<std::string, ggml_backend_buffer_type_t> buft_list;
    if (buft_list.empty()) {
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            auto* dev = ggml_backend_dev_get(i);
            if (auto* buft = ggml_backend_dev_buffer_type(dev)) {
                buft_list[ggml_backend_buft_name(buft)] = buft;
            }
        }
    }

    for (const auto & override : string_split<std::string>(value, ',')) {
        const std::string::size_type pos = override.find('=');
        if (pos == std::string::npos) {
            throw std::invalid_argument("invalid value");
        }
        std::string tensor_name = override.substr(0, pos);
        std::string buffer_type = override.substr(pos + 1);
        if (buft_list.find(buffer_type) == buft_list.end()) {
            printf("Available buffer types:\n");
            for (const auto &[name, type] : buft_list) {
                printf("  %s\n", ggml_backend_buft_name(type));
            }
            throw std::invalid_argument("Attempted to use an invalid buffer override type. Exiting. ");
        }

        leaked_strings.push_back(strdup(tensor_name.c_str()));
        tensor_buft_overrides.push_back({leaked_strings.back(), buft_list.at(buffer_type)});
    }

    if (!tensor_buft_overrides.empty()) {
        //Yes this is some nightmare garbage where it needs a null terminator don't ask me man, this does need to be here.
        tensor_buft_overrides.push_back({nullptr, nullptr});
    }
    return tensor_buft_overrides;
}

llama_model* model_load(
    const char* model_path,
    const int32_t num_gpu_layers,
    const int tensor_split_mode,
    const float* tensor_split,
    const llama_progress_callback callback,
    const char* tensor_type_split_regex,
    const bool use_mmap,
    const bool realtime_process_priority)
{
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = num_gpu_layers;
    model_params.progress_callback = callback;

    model_params.split_mode = static_cast<llama_split_mode>(tensor_split_mode);
    model_params.tensor_split = tensor_split;
    model_params.use_mmap = use_mmap;

    // Requires sudo on unix systems
    // Requires admin for realtime on Windows
    if (realtime_process_priority) {
        set_process_priority(GGML_SCHED_PRIO_REALTIME);
    }

    if (tensor_type_split_regex != nullptr) {
        std::vector<char*> leaked_c_strings;
        const auto overrides = tensor_type_split(std::string(tensor_type_split_regex), leaked_c_strings);

        if (!overrides.empty()) {
            model_params.tensor_buft_overrides = overrides.data();
        }
        llama_model* model = llama_model_load_from_file(model_path, model_params);
        for (char* ptr : leaked_c_strings) {
            free(ptr);
        }
        return model;
    }

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

bool model_vocab_add_bos(const llama_model* model)
{
    return llama_vocab_get_add_bos(&model->vocab);
}

const char* model_vocab_token_to_string(const llama_model* model, const llama_token token) {
    return llama_vocab_get_text(&model->vocab, token);
}

llama_context* ctx_make(
    llama_model* model,
    const unsigned context_length,
    const unsigned num_batches,
    const unsigned num_physical_batches,
    const int32_t num_slots,
    const int32_t num_gpu_layers,
    const int32_t num_threads,
    const bool flash_attn,
    const float rope_freq_base,
    const bool use_yarn,
    int k_cache_quant_type,
    int v_cache_quant_type,
    const float kv_defrag_threshold,
    const bool offload_kqv
) {
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = context_length;
    ctx_params.n_batch = num_batches;
    ctx_params.n_ubatch = num_physical_batches;
    ctx_params.n_seq_max = num_slots;
    ctx_params.no_perf = false;
    ctx_params.flash_attn_type = flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;

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

    // Use only one thread if model is fully offloaded on GPU
    if (num_gpu_layers >= llama_model_n_layer(model) || num_gpu_layers == -1) {
        ctx_params.n_threads = 1;
        ctx_params.n_threads_batch = 1;
    } else {
        ctx_params.n_threads = num_threads;
        ctx_params.n_threads_batch = num_threads;
    }

    ctx_params.type_k = static_cast<ggml_type>(k_cache_quant_type);
    ctx_params.type_v = static_cast<ggml_type>(v_cache_quant_type);
    ctx_params.defrag_thold = kv_defrag_threshold;
    ctx_params.offload_kqv = offload_kqv;
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

llama_memory_t memory_make(llama_context* ctx)
{
    return llama_get_memory(ctx);
}

void memory_clear(llama_memory_t mem)
{
    llama_memory_clear(mem, true);
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

bool has_llguidance() {
    #if defined(LLGUIDANCE_BUILT) || LLGUIDANCE_BUILT != 0
        return true;
    #else
        return false;
    #endif
}
