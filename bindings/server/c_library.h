#ifndef PROCESSOR_INTERFACE_H
#define PROCESSOR_INTERFACE_H

#include "llama.h"

#ifdef __cplusplus

extern "C" {
#endif

    typedef struct Processor Processor;
    typedef struct ReadbackBuffer ReadbackBuffer;
    typedef struct GenerationResources GenerationResources;

    // ~~~ Lcpp Model ~~~

    // LEAKABLE! Ensure you use model_free to clean up.
    llama_model* model_load(
        const char* model_path,
        int32_t num_gpu_layers,
        const int tensor_split_mode,
        const float* tensor_split,
        llama_progress_callback callback,
        const char* tensor_type_split_regex,
        const bool use_mmap,
        const bool realtime_process_priority);

    float model_get_freq_base(
        const llama_model* model);

    void model_free(
        llama_model* model);

    // LEAKABLE! Ensure you use endpoint_free_string to clean up.
    char* model_chat_template(
        const llama_model* model);

    // ~~~ Processor ~~~

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
        const bool add_special);

    bool processor_cancel_work(
        Processor* processor,
        int request_id_to_cancel);

    Processor* processor_make(
        llama_model* model,
        llama_context* ctx,
        llama_memory_t mem,
        int num_processor_slots);

    void processor_free(
        const Processor* processor);

    // ~~~ Lcpp Endpoint ~~~

    // LEAKABLE! Ensure you use endpoint_free_tokens to clean up.
    int32_t* endpoint_tokenize(
        const llama_model* model,
        const char* prompt,
        bool add_special,
        bool parse_special);

    // LEAKABLE! Ensure you use endpoint_free_string to clean up.
    char* endpoint_detokenize(
        const llama_model* model,
        const int32_t* tokens,
        int32_t num_tokens,
        int32_t max_text_size,
        bool add_special,
        bool parse_special);

    void endpoint_free_string(
        const char* str);

    void endpoint_free_tokens(
        const int32_t* tokens);

    // ~~~ Lcpp Vocab ~~~

    llama_token model_vocab_bos(
        const llama_model* model);

    llama_token model_vocab_eos(
        const llama_model* model);

    llama_token model_vocab_eot(
        const llama_model* model);

    bool model_vocab_add_bos(
        const llama_model* model);

    // LEAKABLE! Ensure you use endpoint_free_string to clean up.
    const char* model_vocab_token_to_string(
        const llama_model* model,
        llama_token token);

    // ~~~ Lcpp Context ~~~

    // LEAKABLE! Ensure you use ctx_free to clean up.
    llama_context* ctx_make(
        llama_model* model,
        unsigned context_length,
        unsigned num_batches,
        int32_t num_gpu_layers,
        int32_t num_threads,
        bool flash_attn,
        float rope_freq_base,
        bool use_yarn,
        int k_cache_quant_type,
        int v_cache_quant_type,
        float kv_defrag_threshold,
        bool offload_kqv
    );

    uint32_t ctx_max_seq_len(
        const llama_context* ctx);

    void ctx_free(
        llama_context* ctx);

    llama_memory_t memory_make(
        llama_context* ctx);

    void memory_clear(
        llama_memory_t mem);

    // ~~~ Readback Buffer ~~~

    bool readback_is_buffer_finished(
        ReadbackBuffer* buffer);

    bool readback_read_next(
        ReadbackBuffer* buffer,
        char** outChar,
        llama_token* outToken);

    //TODO::@Z Validate.
    //  Not leakable, owned by readback buffer ?
    char* readback_read_status(
        ReadbackBuffer* buffer);

    void readback_annihilate(
        ReadbackBuffer* buffer);

    // ~~~ Samplers ~~~

    llama_sampler* sampler_dist(
       llama_sampler* chain,
       uint32_t seed);

    llama_sampler* sampler_greedy(
       llama_sampler* chain);

    llama_sampler* sampler_min_p(
       llama_sampler* chain,
       float min_p,
       size_t min_keep);

    llama_sampler* sampler_mirostat_v2(
       llama_sampler* chain,
       uint32_t seed,
       float tau,
       float eta);

    llama_sampler* sampler_penalties(
       llama_sampler* chain,
       int penalty_last_n,
       float penalty_repeat,
       float penalty_freq,
       float penalty_present);

    llama_sampler* sampler_temp(
       llama_sampler* chain,
       float temp);

    llama_sampler* sampler_temp_ext(
       llama_sampler* chain,
       float temp,
       float dynatemp_range,
       float dynatemp_exponent);

    llama_sampler* sampler_top_k(
       llama_sampler* chain,
       int top_k);

    llama_sampler* sampler_top_p(
       llama_sampler* chain,
       float top_p,
       size_t min_keep);

    llama_sampler* sampler_typical(
       llama_sampler* chain,
       float typical_p,
       size_t min_keep);

    llama_sampler* sampler_top_n_sigma(
       llama_sampler* chain,
       float n_sigma);

    llama_sampler* sampler_xtc(
       llama_sampler* chain,
       float xtc_probability,
       float xtc_threshold,
       size_t min_keep,
       uint32_t seed);

    llama_sampler* sampler_grammar(
       llama_sampler* chain,
       const llama_model* model,
       const char* grammar);

    llama_sampler* sampler_dry(
       llama_sampler* chain,
       const llama_model* model,
       float multiplier,
       float base,
       int32_t allowed_length,
       int32_t penalty_last_n,
       const char** sequence_breakers,
       size_t n_breakers);

    llama_sampler* sampler_infill(
       llama_sampler* chain,
       const llama_model* model);

    llama_sampler* sampler_logit_bias(
       llama_sampler* chain,
       const llama_model* model,
       int32_t n_bias,
       const llama_logit_bias* logit_bias);

    llama_sampler* sampler_mirostat(
       llama_sampler* chain,
       const llama_model* model,
       uint32_t seed,
       float tau,
       float eta,
       int m);

    llama_sampler* sampler_llguidance(
        llama_sampler* chain,
        const llama_model* model,
        const char* grammar_data);

    // ~~~ Generation Resources ~~~

    //Leakable! Shared PTR behaviour, use release to free.
    GenerationResources* generation_resources_make();

    void generation_resources_release(
        GenerationResources* resources);

    // ~~~ Features ~~~

    bool has_llguidance();

#ifdef __cplusplus
}
#endif

#endif // PROCESSOR_INTERFACE_H