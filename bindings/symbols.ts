export default {
    // Model functions
    model_load: {
        parameters: [
            "buffer", // model_path: const char*
            "i32", // num_gpu_layers: int32_t
            "buffer", // tensor_split: const float*
            "pointer", // callback: llama_progress_callback
            "buffer", // tensor_type_split_regex: const char*
        ],
        result: "pointer", // llama_model*
        nonblocking: true,
    },

    model_get_freq_base: {
        parameters: ["pointer"], // model: const llama_model*
        result: "f32", // float
    },

    model_free: {
        parameters: ["pointer"], // model: llama_model*
        result: "void",
    },

    model_chat_template: {
        parameters: ["pointer"], // model: const llama_model*
        result: "pointer", // char*
    },

    // Processor functions
    processor_submit_work: {
        parameters: [
            "pointer", // processor: Processor*
            "buffer", // prompt: const char*
            "pointer", // gen_resources: GenerationResources*
            "i32", // max_tokens: int
            "i32", // min_tokens: int
            "u32", // max_slot_n_ctx: unsigned
            "u32", // seed: unsigned
            "buffer", // rewind_strings: const char**
            "u32", // num_rewind_strings: unsigned
            "buffer", // stopping_strings: const char**
            "u32", // num_stopping_strings: unsigned
            "buffer", // stopping_tokens: const int32_t*
            "u32", // num_stopping_tokens: unsigned
            "bool", // add_special: bool
            "bool" // parse_special: bool
        ],
        result: "i32", // int
    },

    processor_cancel_work: {
        parameters: [
            "pointer", // processor: Processor*
            "i32", // request_id_to_cancel: int
        ],
        result: "bool", // bool
    },

    processor_make: {
        parameters: [
            "pointer", // model: llama_model*
            "pointer", // ctx: llama_context*
            "i32", // num_processor_slots: int
        ],
        result: "pointer", // Processor*
        nonblocking: true,
    },

    processor_free: {
        parameters: [
            "pointer", // processor: Processor*
        ],
        result: "void",
    },

    // Endpoint functions
    endpoint_tokenize: {
        parameters: [
            "pointer", // model: const llama_model*
            "buffer", // prompt: const char*
            "bool", // add_special: bool
            "bool", // parse_special: bool
        ],
        result: "pointer", // int32_t*
        nonblocking: true,
    },

    endpoint_detokenize: {
        parameters: [
            "pointer", // model: const llama_model*
            "pointer", // tokens: const int32_t*
            "i32", // num_tokens: int32_t
            "i32", // max_text_size: int32_t
            "bool", // add_special: bool
            "bool", // parse_special: bool
        ],
        result: "pointer", // char*
        nonblocking: true,
    },

    endpoint_free_string: {
        parameters: ["pointer"], // str: const char*
        result: "void",
    },

    endpoint_free_tokens: {
        parameters: ["pointer"], // tokens: const int32_t*
        result: "void",
    },

    // Vocab functions
    model_vocab_bos: {
        parameters: ["pointer"], // model: const llama_model*
        result: "i32", // llama_token
    },

    model_vocab_eos: {
        parameters: ["pointer"], // model: const llama_model*
        result: "i32", // llama_token
    },

    model_vocab_eot: {
        parameters: ["pointer"], // model: const llama_model*
        result: "i32", // llama_token
    },

    model_vocab_token_to_string: {
        parameters: [
            "pointer", // model: const llama_model*
            "i32", // token: llama_token
        ],
        result: "pointer", // const char*
    },

    // Context functions
    ctx_make: {
        parameters: [
            "pointer", // model: llama_model*
            "u32", // cache_size: unsigned
            "u32", // num_batches: unsigned
            "i32", // num_gpu_layers: int32_t
            "i32", // num_threads: int32_t
            "bool", // flash_attn: bool
            "f32", // rope_freq_base: float
            "bool", // use_yarn: bool
            "i32", // k_cache_quant_type: int
            "i32", // v_cache_quant_type: int
            "f32", // kv_defrag_threshold: float
        ],
        result: "pointer", // llama_context*
        nonblocking: true,
    },

    ctx_max_seq_len: {
        parameters: ["pointer"], // ctx: const llama_context*
        result: "u32", // uint32_t
    },

    ctx_free: {
        parameters: ["pointer"], // ctx: llama_context*
        result: "void",
    },

    ctx_clear_kv: {
        parameters: ["pointer"], // ctx: llama_context*
        result: "void",
    },

    readback_is_buffer_finished: {
        parameters: ["pointer"], // buffer: const ReadbackBuffer*
        result: "bool", // bool
    },

    readback_read_next: {
        parameters: [
            "pointer", // buffer: ReadbackBuffer*
            "pointer", // outChar: char**
            "pointer", // outToken: llama_token*
        ],
        result: "bool", // bool
        nonblocking: true,
    },

    readback_read_status: {
        parameters: ["pointer"], // buffer: const ReadbackBuffer*
        result: "pointer", // char*
        nonblocking: true,
    },

    sampler_dist: {
        parameters: [
            "pointer", // chain: llama_sampler*
            "u32", // seed: uint32_t
        ],
        result: "pointer", // llama_sampler*
    },

    sampler_greedy: {
        parameters: ["pointer"], // chain: llama_sampler*
        result: "pointer", // llama_sampler*
    },

    sampler_min_p: {
        parameters: [
            "pointer", // chain: llama_sampler*
            "f32", // min_p: float
            "usize", // min_keep: size_t
        ],
        result: "pointer", // llama_sampler*
    },

    sampler_mirostat_v2: {
        parameters: [
            "pointer", // chain: llama_sampler*
            "u32", // seed: uint32_t
            "f32", // tau: float
            "f32", // eta: float
        ],
        result: "pointer", // llama_sampler*
    },

    sampler_penalties: {
        parameters: [
            "pointer", // chain: llama_sampler*
            "i32", // penalty_last_n: int
            "f32", // penalty_repeat: float
            "f32", // penalty_freq: float
            "f32", // penalty_present: float
        ],
        result: "pointer", // llama_sampler*
    },

    sampler_temp: {
        parameters: [
            "pointer", // chain: llama_sampler*
            "f32", // temp: float
        ],
        result: "pointer", // llama_sampler*
    },

    sampler_temp_ext: {
        parameters: [
            "pointer", // chain: llama_sampler*
            "f32", // temp: float
            "f32", // dynatemp_range: float
            "f32", // dynatemp_exponent: float
        ],
        result: "pointer", // llama_sampler*
    },

    sampler_top_k: {
        parameters: [
            "pointer", // chain: llama_sampler*
            "i32", // top_k: int
        ],
        result: "pointer", // llama_sampler*
    },

    sampler_top_p: {
        parameters: [
            "pointer", // chain: llama_sampler*
            "f32", // top_p: float
            "usize", // min_keep: size_t
        ],
        result: "pointer", // llama_sampler*
    },

    sampler_typical: {
        parameters: [
            "pointer", // chain: llama_sampler*
            "f32", // typical_p: float
            "usize", // min_keep: size_t
        ],
        result: "pointer", // llama_sampler*
    },

    sampler_top_n_sigma: {
        parameters: [
            "pointer", // chain: llama_sampler*
            "f32", // n_sigma: float
        ],
        result: "pointer", // llama_sampler*
    },

    sampler_xtc: {
        parameters: [
            "pointer", // chain: llama_sampler*
            "f32", // xtc_probability: float
            "f32", // xtc_threshold: float
            "usize", // min_keep: size_t
            "u32", // seed: uint32_t
        ],
        result: "pointer", // llama_sampler*
    },

    sampler_grammar: {
        parameters: [
            "pointer", // chain: llama_sampler*
            "pointer", // model: const llama_model*
            "buffer", // grammar: const char*
        ],
        result: "pointer", // llama_sampler*
    },

    sampler_dry: {
        parameters: [
            "pointer", // chain: llama_sampler*
            "pointer", // model: const llama_model*
            "f32", // multiplier: float
            "f32", // base: float
            "i32", // allowed_length: int32_t
            "i32", // penalty_last_n: int32_t
            "pointer", // sequence_breakers: const char**
            "usize", // n_breakers: size_t
        ],
        result: "pointer", // llama_sampler*
    },

    sampler_infill: {
        parameters: [
            "pointer", // chain: llama_sampler*
            "pointer", // model: const llama_model*
        ],
        result: "pointer", // llama_sampler*
    },

    sampler_logit_bias: {
        parameters: [
            "pointer", // chain: llama_sampler*
            "pointer", // model: const llama_model*
            "i32", // n_bias: int32_t
            "pointer", // logit_bias: const llama_logit_bias*
        ],
        result: "pointer", // llama_sampler*
    },

    sampler_mirostat: {
        parameters: [
            "pointer", // chain: llama_sampler*
            "pointer", // model: const llama_model*
            "u32", // seed: uint32_t
            "f32", // tau: float
            "f32", // eta: float
            "i32", // m: int
        ],
        result: "pointer", // llama_sampler*
    },

    sampler_llguidance: {
        parameters: [
            "pointer", // chain: llama_sampler*
            "pointer", // model: const llama_model*
            "buffer", // char*: grammar_data*
        ],
        result: "pointer", // llama_sampler*
    },

    // Generation resources functions
    generation_resources_make: {
        parameters: [],
        result: "pointer",
    },

    generation_resources_release: {
        parameters: [
            "pointer", // GenerationResources* resources
        ],
        result: "void",
    },

    has_llguidance: {
        parameters: [],
        result: "bool",
    },
} as const;
