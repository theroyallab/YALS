#ifndef BINDING_H
#define BINDING_H
#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

    void* LoadModel(const char *modelPath, int numberGpuLayers);
    void* InitiateCtx(void* llamaModel, unsigned contextLength, unsigned numBatches);

    void* CreateReadbackBuffer();
    void* ReadbackNext(void* readbackBufferPtr);
    void WriteToReadbackBuffer(void* readbackBufferPtr, char* stringData);
    bool IsReadbackBufferDone(void* readbackBufferPtr);

    /* SAMPLERS
     * */
    void* MakeSampler();
    void* DistSampler(void* sampler, uint32_t seed);
    void* GrammarSampler(void* sampler, const llama_model* model, const char* grammar, const char* root);
    void* GreedySampler(void* sampler);
    void* InfillSampler(void* sampler, const llama_model* model);
    void* LogitBiasSampler(void* sampler, const llama_model* model, size_t nBias, const llama_logit_bias* logitBias);
    void* MinPSampler(void* sampler, float minP, size_t minKeep);
    void* MirostatSampler(void* sampler, int nVocab, uint32_t seed, float tau, float eta, int m);
    void* MirostatV2Sampler(void* sampler, uint32_t seed, float tau, float eta);
    void* PenaltiesSampler(void* sampler, int nVocab, llama_token eosToken, llama_token nlToken, int penaltyLastN, float penaltyRepeat, float penaltyFreq, float penaltyPresent, bool penalizeNl, bool ignoreEos);
    void* SoftmaxSampler(void* sampler);
    void* TailFreeSampler(void* sampler, float z, size_t minKeep);
    void* TempSampler(void* sampler, float temp);
    void* TempExtSampler(void* sampler, float temp, float dynatempRange, float dynatempExponent);
    void* TopKSampler(void* sampler, int topK);
    void* TopPSampler(void* sampler, float topP, size_t minKeep);
    void* TypicalSampler(void* sampler, float typicalP, size_t minKeep);
    void* XtcSampler(void* sampler, float xtcProbability, float xtcThreshold, size_t minKeep, uint32_t seed);
    /* SAMPLERS
     * */

    void Infer(
        void* llamaModelPtr,
        void* samplerPtr,
        void* contextPtr,
        const char *prompt,
        unsigned numberTokensToPredict);

    void InferToReadbackBuffer(
        void* llamaModelPtr,
        void* samplerPtr,
        void* contextPtr,
        void* readbackBufferPtr,
        const char *prompt,
        const unsigned numberTokensToPredict);

#ifdef __cplusplus
}
#endif
#endif // BINDING_H
