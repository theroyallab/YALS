export default {
    LoadModel: {
        parameters: [
            "buffer", // const char *modelPath
            "i32", // int numberGpuLayers
            "pointer", // llama_progress_callback callback
        ],
        result: "pointer" as const, // void*
        nonblocking: true,
    },
    InitiateCtx: {
        parameters: [
            "pointer", // void* llamaModel
            "u32", // unsigned contextLength
            "u32", // unsigned numBatches
            "bool", // bool flashAttn
            "f32", // float ropeFreqBase
            "f32", // float ropeFreqScale
            "i32", // ggml_type kCacheQuantType
            "i32", // ggml_type vCacheQuantType
        ],
        result: "pointer" as const, // void*
        nonblocking: true,
    },
    BosToken: {
        parameters: [
            "pointer", // llama_model* model
        ],
        result: "i32" as const, // llama_token
    },
    EosToken: {
        parameters: [
            "pointer", // llama_model* model
        ],
        result: "i32" as const, // llama_token
    },
    EotToken: {
        parameters: [
            "pointer", // llama_model* model
        ],
        result: "i32" as const, // llama_token
    },
    FreeSampler: {
        parameters: ["pointer"],
        result: "void",
        nonblocking: true,
    },
    FreeCtx: {
        parameters: ["pointer"],
        result: "void",
        nonblocking: true,
    },
    ClearContextKVCache: {
        parameters: ["pointer"],
        result: "void",
    },
    FreeModel: {
        parameters: ["pointer"],
        result: "void",
        nonblocking: true,
    },
    TokenToString: {
        parameters: ["pointer", "i32"],
        result: "pointer" as const,
        nonblocking: true,
    },
    MakeSampler: {
        parameters: [],
        result: "pointer" as const, // void*
    },
    DistSampler: {
        parameters: ["pointer", "u32"], // void* sampler, uint32_t seed
        result: "pointer" as const, // void*
    },
    GrammarSampler: {
        parameters: ["pointer", "pointer", "buffer", "buffer"], // void* sampler, const llama_model* model, const char* grammar, const char* root
        result: "pointer" as const, // void*
    },
    GreedySampler: {
        parameters: ["pointer"], // void* sampler
        result: "pointer" as const, // void*
    },
    InfillSampler: {
        parameters: ["pointer", "pointer"], // void* sampler, const llama_model* model
        result: "pointer" as const, // void*
    },
    LogitBiasSampler: {
        parameters: ["pointer", "pointer", "i32", "pointer"], // void* sampler, const llama_model* model, int32_t nBias, const llama_logit_bias* logitBias
        result: "pointer" as const, // void*
    },
    MinPSampler: {
        parameters: ["pointer", "f32", "usize"], // void* sampler, float minP, size_t minKeep
        result: "pointer" as const, // void*
    },
    MirostatSampler: {
        parameters: ["pointer", "pointer", "u32", "f32", "f32", "i32"], // void* sampler, void* llamaModel, uint32_t seed, float tau, float eta, int m
        result: "pointer" as const, // void*
    },
    MirostatV2Sampler: {
        parameters: ["pointer", "u32", "f32", "f32"], // void* sampler, uint32_t seed, float tau, float eta
        result: "pointer" as const, // void*
    },
    PenaltiesSampler: {
        parameters: [
            "pointer",
            "i32",
            "f32",
            "f32",
            "f32",
        ], // void* sampler, int penaltyLastN, float penaltyRepeat, float penaltyFreq, float penaltyPresent
        result: "pointer" as const, // void*
    },
    TempSampler: {
        parameters: ["pointer", "f32"], // void* sampler, float temp
        result: "pointer" as const, // void*
    },
    TempExtSampler: {
        parameters: ["pointer", "f32", "f32", "f32"], // void* sampler, float temp, float dynatempRange, float dynatempExponent
        result: "pointer" as const, // void*
    },
    TopKSampler: {
        parameters: ["pointer", "i32"], // void* sampler, int topK
        result: "pointer" as const, // void*
    },
    TopPSampler: {
        parameters: ["pointer", "f32", "usize"], // void* sampler, float topP, size_t minKeep
        result: "pointer" as const, // void*
    },
    TypicalSampler: {
        parameters: ["pointer", "f32", "usize"], // void* sampler, float typicalP, size_t minKeep
        result: "pointer" as const, // void*
    },
    XtcSampler: {
        parameters: ["pointer", "f32", "f32", "usize", "u32"], // void* sampler, float xtcProbability, float xtcThreshold, size_t minKeep, uint32_t seed
        result: "pointer" as const, // void*
    },
    DrySampler: {
        parameters: [
            "pointer",
            "pointer",
            "f32",
            "f32",
            "i32",
            "i32",
            "buffer",
            "u64",
        ], // void* sampler, const llama_model* model, float multiplier, float base, int32_t allowed_length, int32_t penalty_last_n, const char* const* sequence_breakers, size_t n_breakers
        result: "pointer" as const, // void*
    },
    CreateReadbackBuffer: {
        parameters: [],
        result: "pointer" as const, // void*
    },
    ResetReadbackBuffer: {
        parameters: ["pointer"], // ReadbackBuffer*
        result: "void",
    },
    ReadbackNext: {
        parameters: ["pointer"], // void* readbackBufferPtr
        result: "pointer" as const, // void*
        nonblocking: true,
    },
    IsReadbackBufferDone: {
        parameters: ["pointer"], // void* readbackBufferPtr
        result: "bool" as const,
    },
    InferToReadbackBuffer: {
        parameters: [
            "pointer", // const llama_model* model
            "pointer", // llama_sampler* sampler
            "pointer", // llama_context* context
            "pointer", // ReadbackBuffer* readbackBufferPtr
            "buffer", // const char* prompt
            "u32", // const unsigned numberTokensToPredict
            "bool", // const bool addSpecial
            "bool", // const bool parseSpecial
            "pointer", // ggml_abort_callback abortCallback
            "buffer", // const char** rewindStrings
            "u32", // count of rewind strings
            "buffer", // const char** stoppingStrings
            "u32", // count of stop strings
        ],
        result: "pointer" as const,
        nonblocking: true,
    },
} as const;
