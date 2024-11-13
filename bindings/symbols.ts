export default {
    LoadModel: {
        parameters: [
            "pointer", // const char *modelPath
            "i32", // int numberGpuLayers
        ],
        result: "pointer" as const, // void*
        nonblocking: true,
    },
    InitiateCtx: {
        parameters: [
            "pointer", // void* llamaModel
            "u32", // unsigned contextLength
            "u32", // unsigned numBatches
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
    FreeModel: {
        parameters: ["pointer"],
        result: "void",
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
        parameters: ["pointer", "pointer", "pointer", "pointer"], // void* sampler, const llama_model* model, const char* grammar, const char* root
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
            "i32",
            "i32",
            "i32",
            "f32",
            "f32",
            "f32",
            "bool",
            "bool",
        ], // void* sampler, int nVocab, llama_token eosToken, llama_token nlToken, int penaltyLastN, float penaltyRepeat, float penaltyFreq, float penaltyPresent, bool penalizeNl, bool ignoreEos
        result: "pointer" as const, // void*
    },
    TailFreeSampler: {
        parameters: ["pointer", "f32", "usize"], // void* sampler, float z, size_t minKeep
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
            "pointer",
            "u64",
        ], // void* sampler, const llama_model* model, float multiplier, float base, int32_t allowed_length, int32_t penalty_last_n, const char* const* sequence_breakers, size_t n_breakers
        result: "pointer" as const, // void*
    },
    CreateReadbackBuffer: {
        parameters: [],
        result: "pointer" as const, // void*
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
    InferChat: {
        parameters: [
            "pointer", // const llama_model* model
            "pointer", // llama_sampler* sampler
            "pointer", // llama_context* context
            "pointer", // ReadbackBuffer* readbackBufferPtr
            "pointer", // const char* nextMessage
            "u32",    // const unsigned numberTokensToPredict
        ],
        result: "void" as const,
        nonblocking: true,
    },
    InferToReadbackBuffer: {
        parameters: [
            "pointer", // const llama_model* model
            "pointer", // llama_sampler* sampler
            "pointer", // llama_context* context
            "pointer", // ReadbackBuffer* readbackBufferPtr
            "pointer", // const char* prompt
            "u32",    // const unsigned numberTokensToPredict
        ],
        result: "pointer" as const,
        nonblocking: true,
    },
} as const;
