#include "binding.h"

#include <iostream>
#include <optional>
#include <vector>
#include <cstring>

void* LoadModel(const char *modelPath, int numberGpuLayers)
{
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = numberGpuLayers;

    llama_model* model = llama_load_model_from_file(modelPath, model_params);

    llama_add_bos_token(model);
    llama_add_eos_token(model);

    return model;
}

struct ReadbackBuffer
{
    unsigned lastReadbackIndex {0};
    bool done {false};
    std::vector<char*>* data = new std::vector<char*>();
};

bool IsReadbackBufferDone(void* readbackBufferPtr)
{
    return static_cast<ReadbackBuffer*>(readbackBufferPtr)->done;
}

void* CreateReadbackBuffer()
{
    return new ReadbackBuffer {};
}

void WriteToReadbackBuffer(void* readbackBufferPtr, char* stringData)
{
    auto* buffer = static_cast<ReadbackBuffer*>(readbackBufferPtr);
    buffer->data->push_back(stringData);
}

void* ReadbackNext(void* readbackBufferPtr)
{
    auto* buffer = static_cast<ReadbackBuffer*>(readbackBufferPtr);

    //we're racing faster than writes or at the end of the buffer.
    if (buffer->lastReadbackIndex >= buffer->data->size())
    {
        return nullptr;
    }

    char* stringPtr = buffer->data->at(buffer->lastReadbackIndex);
    buffer->lastReadbackIndex++;

    return stringPtr;
}

void* InitiateCtx(void* llamaModel, const unsigned contextLength, const unsigned numBatches)
{
    auto* model = static_cast<llama_model*>(llamaModel);
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = contextLength;
    ctx_params.n_batch = numBatches;
    ctx_params.no_perf = false;
    llama_context* ctx = llama_new_context_with_model(model, ctx_params);

    if (ctx == nullptr) {
        std::cerr << "error: couldn't make llama ctx in InitiateCtx()" << std::endl;
        return nullptr;
    }

    return ctx;
}

void* MakeSampler()
{
    llama_sampler_chain_params lparams = llama_sampler_chain_default_params();
    lparams.no_perf = false;
    const auto sampler = llama_sampler_chain_init(lparams);
    return sampler;
}

// Independent of order
void* DistSampler(void* sampler, uint32_t seed)
{
    llama_sampler_chain_add(static_cast<llama_sampler*>(sampler), llama_sampler_init_dist(seed));
    return sampler;
}

// Independent of order
void* GrammarSampler(void* sampler, const llama_model* model, const char* grammar, const char* root)
{
    llama_sampler_chain_add(static_cast<llama_sampler*>(sampler), llama_sampler_init_grammar(model, grammar, root));
    return sampler;
}

// Typically used as the last sampler in the chain
void* GreedySampler(void* sampler)
{
    llama_sampler_chain_add(static_cast<llama_sampler*>(sampler), llama_sampler_init_greedy());
    return sampler;
}

// Independent of order
void* InfillSampler(void* sampler, const llama_model* model)
{
    llama_sampler_chain_add(static_cast<llama_sampler*>(sampler), llama_sampler_init_infill(model));
    return sampler;
}

// Typically applied early in the sampling chain
void* LogitBiasSampler(void* sampler, const llama_model* model, size_t nBias, const llama_logit_bias* logitBias)
{
    llama_sampler_chain_add(
        static_cast<llama_sampler*>(sampler),
        llama_sampler_init_logit_bias(
            llama_n_vocab(model),
            nBias,
            logitBias
        )
    );
    return sampler;
}

// Independent of order, but typically applied after topK or topP
void* MinPSampler(void* sampler, float minP, size_t minKeep)
{
    llama_sampler_chain_add(static_cast<llama_sampler*>(sampler), llama_sampler_init_min_p(minP, minKeep));
    return sampler;
}

// Depends on temperature, should be applied after tempSampler
void* MirostatSampler(void* sampler, int nVocab, uint32_t seed, float tau, float eta, int m)
{
    llama_sampler_chain_add(static_cast<llama_sampler*>(sampler), llama_sampler_init_mirostat(nVocab, seed, tau, eta, m));
    return sampler;
}

// Depends on temperature, should be applied after tempSampler
void* MirostatV2Sampler(void* sampler, uint32_t seed, float tau, float eta)
{
    llama_sampler_chain_add(static_cast<llama_sampler*>(sampler), llama_sampler_init_mirostat_v2(seed, tau, eta));
    return sampler;
}

// Typically applied early in the sampling chain
void* PenaltiesSampler(void* sampler, int nVocab, llama_token eosToken, llama_token nlToken, int penaltyLastN, float penaltyRepeat, float penaltyFreq, float penaltyPresent, bool penalizeNl, bool ignoreEos)
{
    llama_sampler_chain_add(static_cast<llama_sampler*>(sampler), llama_sampler_init_penalties(nVocab, eosToken, nlToken, penaltyLastN, penaltyRepeat, penaltyFreq, penaltyPresent, penalizeNl, ignoreEos));
    return sampler;
}

// Typically applied after other sampling methods and before distSampler
void* SoftmaxSampler(void* sampler)
{
    llama_sampler_chain_add(static_cast<llama_sampler*>(sampler), llama_sampler_init_softmax());
    return sampler;
}

// Independent of order, but typically applied after topK or topP
void* TailFreeSampler(void* sampler, float z, size_t minKeep)
{
    llama_sampler_chain_add(static_cast<llama_sampler*>(sampler), llama_sampler_init_tail_free(z, minKeep));
    return sampler;
}

// Typically applied early in the sampling chain
void* TempSampler(void* sampler, float temp)
{
    llama_sampler_chain_add(static_cast<llama_sampler*>(sampler), llama_sampler_init_temp(temp));
    return sampler;
}

// Typically applied early in the sampling chain
void* TempExtSampler(void* sampler, float temp, float dynatempRange, float dynatempExponent)
{
    llama_sampler_chain_add(static_cast<llama_sampler*>(sampler), llama_sampler_init_temp_ext(temp, dynatempRange, dynatempExponent));
    return sampler;
}

// Typically applied early in the sampling chain
void* TopKSampler(void* sampler, int topK)
{
    llama_sampler_chain_add(static_cast<llama_sampler*>(sampler), llama_sampler_init_top_k(topK));
    return sampler;
}

// Typically applied after topKSampler
void* TopPSampler(void* sampler, float topP, size_t minKeep)
{
    llama_sampler_chain_add(static_cast<llama_sampler*>(sampler), llama_sampler_init_top_p(topP, minKeep));
    return sampler;
}

// Independent of order, but typically applied after topK or topP
void* TypicalSampler(void* sampler, float typicalP, size_t minKeep)
{
    llama_sampler_chain_add(static_cast<llama_sampler*>(sampler), llama_sampler_init_typical(typicalP, minKeep));
    return sampler;
}

// Independent of order
void* XtcSampler(void* sampler, float xtcProbability, float xtcThreshold, size_t minKeep, uint32_t seed)
{
    llama_sampler_chain_add(static_cast<llama_sampler*>(sampler), llama_sampler_init_xtc(xtcProbability, xtcThreshold, minKeep, seed));
    return sampler;
}

std::optional<std::string> TokenToPiece(const llama_model* llamaModel, const unsigned id)
{
    char buf[128];
    int n = llama_token_to_piece(llamaModel, id, buf, sizeof(buf), 0, true);
    if (n < 0) {
        std::cerr << "error: failed to convert token to piece in TokenToPiece()" << std::endl;
        return std::nullopt;
    }
    return std::string{buf, static_cast<size_t>(n)};
}

std::optional<std::vector<llama_token>> TokenizePrompt(const llama_model* llamaModel, const std::string_view& prompt)
{
    const int n_prompt = -llama_tokenize(llamaModel, prompt.data(), prompt.size(), nullptr, 0, true, true);
    std::vector<llama_token> tokenizedPrompt(n_prompt);
    if (llama_tokenize(llamaModel, prompt.data(), prompt.size(), tokenizedPrompt.data(), tokenizedPrompt.size(), true, true) < 0) {
        std::cerr << "error: failed to tokenize the prompt in TokenizePrompt()" << std::endl;
        return std::nullopt;
    }
    return tokenizedPrompt;
}

void Infer(
    void* llamaModelPtr,
    void* samplerPtr,
    void* contextPtr,
    const char *prompt,
    const unsigned numberTokensToPredict)
{
    const auto llamaModel = static_cast<llama_model*>(llamaModelPtr);
    const auto sampler = static_cast<llama_sampler*>(samplerPtr);
    const auto context = static_cast<llama_context*>(contextPtr);

    auto promptTokens = TokenizePrompt(llamaModel, prompt).value();

    const int numTokensToGenerate = (promptTokens.size() - 1) + numberTokensToPredict;
    llama_batch batch = llama_batch_get_one(promptTokens.data(), promptTokens.size());

    int nDecode = 0;
    llama_token newTokenId;
    //inference
    for (int tokenPosition = 0; tokenPosition + batch.n_tokens < numTokensToGenerate; tokenPosition += batch.n_tokens) {
        // evaluate the current batch with the transformer model
        if (llama_decode(context, batch)) {
            std::cerr << "error: failed to eval, return code 1 in Infer()" << std::endl;
            return;
        }

        // sample the next token
        {
            newTokenId = llama_sampler_sample(sampler, context, -1);

            // is it an end of generation?
            if (llama_token_is_eog(llamaModel, newTokenId)) {
                break;
            }

            std::cout << TokenToPiece(llamaModel, newTokenId).value();
            std::flush(std::cout);

            // prepare the next batch with the sampled token
            batch = llama_batch_get_one(&newTokenId, 1);

            nDecode += 1;
        }
    }
    llama_perf_sampler_print(sampler);
    llama_perf_context_print(context);
}

void InferToReadbackBuffer(
    void* llamaModelPtr,
    void* samplerPtr,
    void* contextPtr,
    void* readbackBufferPtr,
    const char *prompt,
    const unsigned numberTokensToPredict)
{
    const auto llamaModel = static_cast<llama_model*>(llamaModelPtr);
    const auto sampler = static_cast<llama_sampler*>(samplerPtr);
    const auto context = static_cast<llama_context*>(contextPtr);

    auto promptTokens = TokenizePrompt(llamaModel, prompt).value();

    const int numTokensToGenerate = (promptTokens.size() - 1) + numberTokensToPredict;
    llama_batch batch = llama_batch_get_one(promptTokens.data(), promptTokens.size());

    llama_token newTokenId;
    int tokenPosition = 0;
    //inference
    for (tokenPosition = 0; tokenPosition + batch.n_tokens < numTokensToGenerate; tokenPosition += batch.n_tokens ) {
        // evaluate the current batch with the transformer model
        if (llama_decode(context, batch)) {
            std::cerr << "error: failed to eval, return code 1 in Infer()" << std::endl;
            return;
        }

        // sample the next token
        {
            newTokenId = llama_sampler_sample(sampler, context, -1);

            // is it an end of generation?
            if (llama_token_is_eog(llamaModel, newTokenId)) {
                break;
            }

            auto piece = TokenToPiece(llamaModel, newTokenId).value();
            WriteToReadbackBuffer(readbackBufferPtr, strdup(piece.c_str()));

            // prepare the next batch with the sampled token
            batch = llama_batch_get_one(&newTokenId, 1);
        }
    }
    static_cast<ReadbackBuffer*>(readbackBufferPtr)->done = true;

    llama_perf_sampler_print(sampler);
    llama_perf_context_print(context);
}

void FreeSampler(llama_sampler* sampler)
{
    llama_sampler_free(sampler);
}

void FreeCtx(llama_context* ctx)
{
    llama_free(ctx);
}

void FreeModel(llama_model* model)
{
    llama_free_model(model);
}