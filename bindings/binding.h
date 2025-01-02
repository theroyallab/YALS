#ifndef BINDING_H
#define BINDING_H
#include "llama.h"

#ifdef __cplusplus

//Array of strings passed via FFI, equiv to const char[]*
typedef struct {
    const char** strings;
    size_t count;
} FfiStringArray;

extern "C" {
#endif
    struct ReadbackBuffer;

    void TestPrint(const char* text);
    llama_model* LoadModel(const char* modelPath, int numberGpuLayers, llama_progress_callback callback);
    llama_context* InitiateCtx(llama_model *model, unsigned contextLength, unsigned numBatches,
        bool flashAttn, float ropeFreqBase, float ropeFreqScale);
    llama_token BosToken(const llama_model* model);
    llama_token EosToken(const llama_model* model);
    llama_token EotToken(const llama_model* model);
    void FreeSampler(llama_sampler* sampler);
    void FreeModel(llama_model* model);
    void FreeCtx(llama_context* ctx);
    void ClearContextKVCache(llama_context* ctx);
    const char* TokenToString(const llama_model* model, const llama_token token);

    ReadbackBuffer* CreateReadbackBuffer();
    void ResetReadbackBuffer(ReadbackBuffer* buffer);

    char* ReadbackNext(ReadbackBuffer *buffer);
    void WriteToReadbackBuffer(const ReadbackBuffer* buffer, char* stringData);
    bool IsReadbackBufferDone(const ReadbackBuffer* buffer);

    /* SAMPLERS
     * */
    llama_sampler* MakeSampler();
    llama_sampler* DistSampler(llama_sampler* sampler, uint32_t seed);
    llama_sampler* GrammarSampler(
        llama_sampler* sampler, const llama_model* model, const char* grammar, const char* root);
    llama_sampler* GreedySampler(llama_sampler* sampler);
    llama_sampler* InfillSampler(llama_sampler* sampler, const llama_model* model);
    llama_sampler* LogitBiasSampler(
        llama_sampler* sampler, const llama_model* model, int32_t nBias, const llama_logit_bias* logitBias);
    llama_sampler* MinPSampler(llama_sampler* sampler, float minP, size_t minKeep);
    llama_sampler* MirostatSampler(
        llama_sampler* sampler, const llama_model* model, uint32_t seed, float tau, float eta, int m);
    llama_sampler* MirostatV2Sampler(llama_sampler* sampler, uint32_t seed, float tau, float eta);
    llama_sampler* PenaltiesSampler(
        llama_sampler* sampler, int penaltyLastN, float penaltyRepeat,
        float penaltyFreq, float penaltyPresent);
    llama_sampler* TempSampler(llama_sampler* sampler, float temp);
    llama_sampler* TempExtSampler(llama_sampler* sampler, float temp, float dynatempRange, float dynatempExponent);
    llama_sampler* TopKSampler(llama_sampler* sampler, int topK);
    llama_sampler* TopPSampler(llama_sampler* sampler, float topP, size_t minKeep);
    llama_sampler* TypicalSampler(llama_sampler* sampler, float typicalP, size_t minKeep);
    llama_sampler* XtcSampler(
        llama_sampler* sampler, float xtcProbability, float xtcThreshold, size_t minKeep, uint32_t seed);
    llama_sampler* DrySampler(
        llama_sampler* sampler, const llama_model* model, float multiplier, float base, int32_t allowed_length,
        int32_t penalty_last_n, const char** sequence_breakers, size_t n_breakers);
    /* SAMPLERS
        * */

    const char* InferToReadbackBuffer(
        const llama_model* model,
        llama_sampler* sampler,
        llama_context* context,
        ReadbackBuffer* readbackBufferPtr,
        const char* prompt,
        const unsigned numberTokensToPredict,
        const bool addSpecial,
        const bool parseSpecial,
        ggml_abort_callback abortCallback,
        const char** rewindStrings,
        const unsigned numRewindStrings,
        const char** stoppingStrings,
        const unsigned numStoppingStrings);

#ifdef __cplusplus
}
#endif
#endif // BINDING_H
