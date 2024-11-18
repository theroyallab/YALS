#include "binding.h"
#include "trie.hpp"
#include <iostream>
#include <optional>
#include <cstring>
#include <iomanip>
#include <vector>

void TestPrint(const char* text)
{
    std::cout << text << std::endl;
}

llama_model* LoadModel(const char* modelPath, int numberGpuLayers)
{
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = numberGpuLayers;

    llama_model* model = llama_load_model_from_file(modelPath, model_params);

    return model;
}

llama_context* InitiateCtx(llama_model* model, const unsigned contextLength, const unsigned numBatches)
{
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

llama_token BosToken(const llama_model* model)
{
    return llama_token_bos(model);
}

llama_token EosToken(const llama_model* model)
{
    return llama_token_eos(model);
}

llama_token EotToken(const llama_model* model)
{
    return llama_token_eot(model);
}

const char* TokenToString(const llama_model* model, const llama_token token) {
    return llama_token_get_text(model, token);
}

void FreeSampler(llama_sampler* sampler)
{
    llama_sampler_free(sampler);
}

void FreeCtx(llama_context* ctx)
{
    llama_free(ctx);
}

void ClearContextKVCache(llama_context* ctx)
{
    llama_kv_cache_clear(ctx);
}

void FreeModel(llama_model* model)
{
    llama_free_model(model);
}

void PrintPerformanceInfo(const llama_context* context) {
    const auto data = llama_perf_context(context);

    const double prompt_tok_per_sec = 1e3 / data.t_p_eval_ms * data.n_p_eval;
    const double gen_tok_per_sec = 1e3 / data.t_eval_ms * data.n_eval;

    std::cout << "\n\n" << std::fixed << std::setprecision(2)
              << "Prompt Processing: " << prompt_tok_per_sec << " tok/s, "
              << "Text Generation: " << gen_tok_per_sec << " tok/s" << "\n" << std::endl;
}

struct ReadbackBuffer
{
    unsigned lastReadbackIndex {0};
    bool done {false};
    std::vector<char*>* data = new std::vector<char*>();
};

bool IsReadbackBufferDone(const ReadbackBuffer* buffer)
{
    return buffer->done;
}

ReadbackBuffer* CreateReadbackBuffer()
{
    return new ReadbackBuffer {};
}

void WriteToReadbackBuffer(const ReadbackBuffer* buffer, char* stringData)
{
    buffer->data->push_back(stringData);
}

char* ReadbackNext(ReadbackBuffer* buffer)
{
    //we're racing faster than writes or at the end of the buffer.
    if (buffer->lastReadbackIndex >= buffer->data->size())
    {
        return nullptr;
    }

    char* stringPtr = buffer->data->at(buffer->lastReadbackIndex);
    buffer->lastReadbackIndex++;

    return stringPtr;
}

llama_sampler* MakeSampler()
{
    llama_sampler_chain_params lparams = llama_sampler_chain_default_params();
    lparams.no_perf = false;
    const auto sampler = llama_sampler_chain_init(lparams);
    return sampler;
}

// Independent of order
llama_sampler* DistSampler(llama_sampler* sampler, const uint32_t seed)
{
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(seed));
    return sampler;
}

// Independent of order
llama_sampler* GrammarSampler(llama_sampler* sampler, const llama_model* model, const char* grammar, const char* root)
{
    llama_sampler_chain_add(sampler, llama_sampler_init_grammar(model, grammar, root));
    return sampler;
}

llama_sampler* DrySampler(llama_sampler* sampler, const llama_model* model, const float multiplier,
                          const float base, const int32_t allowed_length, const int32_t penalty_last_n,
                          const char** sequence_breakers, const size_t n_breakers)
{
    llama_sampler_chain_add(
        sampler,
        llama_sampler_init_dry(
            model, multiplier, base, allowed_length,
            penalty_last_n, sequence_breakers, n_breakers
        )
    );
    return sampler;
}

// Typically used as the last sampler in the chain
llama_sampler* GreedySampler(llama_sampler* sampler)
{
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
    return sampler;
}

// Independent of order
llama_sampler* InfillSampler(llama_sampler* sampler, const llama_model* model)
{
    llama_sampler_chain_add(sampler, llama_sampler_init_infill(model));
    return sampler;
}

// Typically applied early in the sampling chain
llama_sampler* LogitBiasSampler(
    llama_sampler* sampler, const llama_model* model, const int32_t nBias, const llama_logit_bias* logitBias)
{
    llama_sampler_chain_add(
        sampler,
        llama_sampler_init_logit_bias(
            llama_n_vocab(model),
            nBias,
            logitBias
        )
    );
    return sampler;
}

// Independent of order, but typically applied after topK or topP
llama_sampler* MinPSampler(llama_sampler* sampler, const float minP, const size_t minKeep)
{
    llama_sampler_chain_add(sampler, llama_sampler_init_min_p(minP, minKeep));
    return sampler;
}

// Depends on temperature, should be applied after tempSampler
llama_sampler* MirostatSampler(
    llama_sampler* sampler, const llama_model* model, const uint32_t seed,
    const float tau, const float eta, const int m)
{
    const int nVocab = llama_n_vocab(model);
    llama_sampler_chain_add(sampler, llama_sampler_init_mirostat(nVocab, seed, tau, eta, m));
    return sampler;
}

// Depends on temperature, should be applied after tempSampler
llama_sampler* MirostatV2Sampler(llama_sampler* sampler, const uint32_t seed, const float tau, const float eta)
{
    llama_sampler_chain_add(sampler, llama_sampler_init_mirostat_v2(seed, tau, eta));
    return sampler;
}

// Typically applied early in the sampling chain
llama_sampler* PenaltiesSampler(llama_sampler* sampler, const llama_model* model,
                                const int penaltyLastN, const float penaltyRepeat,
                                const float penaltyFreq, const float penaltyPresent, const bool penalizeNl,
                                const bool ignoreEos)
{
    const int nVocab = llama_n_vocab(model);
    llama_sampler_chain_add(sampler, llama_sampler_init_penalties(
        nVocab, LLAMA_TOKEN_NULL, llama_token_nl(model), penaltyLastN,
        penaltyRepeat, penaltyFreq, penaltyPresent, penalizeNl, ignoreEos));
    return sampler;
}

// Typically applied early in the sampling chain
llama_sampler* TempSampler(llama_sampler* sampler, const float temp)
{
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(temp));
    return sampler;
}

// Typically applied early in the sampling chain
llama_sampler* TempExtSampler(
    llama_sampler* sampler, const float temp, const float dynatempRange, const float dynatempExponent)
{
    llama_sampler_chain_add(sampler, llama_sampler_init_temp_ext(temp, dynatempRange, dynatempExponent));
    return sampler;
}

// Typically applied early in the sampling chain
llama_sampler* TopKSampler(llama_sampler* sampler, const int topK)
{
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(topK));
    return sampler;
}

// Typically applied after topKSampler
llama_sampler* TopPSampler(llama_sampler* sampler, const float topP, const size_t minKeep)
{
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(topP, minKeep));
    return sampler;
}

// Independent of order, but typically applied after topK or topP
llama_sampler* TypicalSampler(llama_sampler* sampler, const float typicalP, const size_t minKeep)
{
    llama_sampler_chain_add(sampler, llama_sampler_init_typical(typicalP, minKeep));
    return sampler;
}

// Independent of order
llama_sampler* XtcSampler(
    llama_sampler* sampler, const float xtcProbability, const float xtcThreshold,
    const size_t minKeep, const uint32_t seed)
{
    llama_sampler_chain_add(sampler, llama_sampler_init_xtc(xtcProbability, xtcThreshold, minKeep, seed));
    return sampler;
}

std::optional<std::string> TokenToPiece(const llama_model* llamaModel, const llama_token id)
{
    char buf[128];
    const int n = llama_token_to_piece(llamaModel, id, buf, sizeof(buf), 0, true);
    if (n < 0) {
        std::cerr << "error: failed to convert token to piece in TokenToPiece()" << std::endl;
        return std::nullopt;
    }

    return std::string{buf, static_cast<size_t>(n)};
}

std::optional<std::vector<llama_token>> Tokenize(
    const llama_model* llamaModel, const llama_context* context, const std::string_view& prompt,
    const bool addSpecial, const bool parseSpecial) {

    const int n_prompt = -llama_tokenize(llamaModel, prompt.data(), prompt.size(),
                                       nullptr, 0, addSpecial, parseSpecial);
    std::vector<llama_token> tokenizedPrompt(n_prompt);
    
    bool add_bos = (llama_get_kv_cache_used_cells(context) == 0) & addSpecial;

    if (llama_tokenize(llamaModel, prompt.data(), prompt.size(),
        tokenizedPrompt.data(), tokenizedPrompt.size(),
        add_bos, parseSpecial) < 0) {
        std::cerr << "error: failed to tokenize the prompt in TokenizePrompt()" << std::endl;
        return std::nullopt;
    }

    return tokenizedPrompt;
}

const char* InferToReadbackBuffer(
    const llama_model* model,
    llama_sampler* sampler,
    llama_context* context,
    ReadbackBuffer* readbackBufferPtr,
    const char* prompt,
    const unsigned numberTokensToPredict,
    const bool addSpecial,
    const bool parseSpecial,
    const char** rewindStrings,
    const unsigned numRewindStrings,
    const char** stoppingStrings,
    const unsigned numStoppingStrings)
{
    auto promptTokens = Tokenize(model, context, prompt, addSpecial, parseSpecial).value();

    const int numTokensToGenerate = (promptTokens.size() - 1) + numberTokensToPredict;
    llama_batch firstBatch = llama_batch_get_one(promptTokens.data(), promptTokens.size());

    MatchTrie::MatchTrie matchingTrie;

    matchingTrie.AddMatchableWords(rewindStrings, numRewindStrings, MatchTrie::MatchType::REWIND);
    matchingTrie.AddMatchableWords(stoppingStrings, numStoppingStrings, MatchTrie::MatchType::STOP);

    std::string response;
    std::string buffer;

    auto gen = [&](const llama_batch& batch) -> std::pair<llama_token, bool> {
        int n_ctx = llama_n_ctx(context);
        int n_ctx_used = llama_get_kv_cache_used_cells(context);
        if (n_ctx_used + batch.n_tokens > n_ctx) {
            std::cerr << "Context size exceeded, must abort." << std::endl;
            return {0, true};
        }

        if (llama_decode(context, batch)) {
            std::cerr << "error: failed to eval, return code 1 in Infer()" << std::endl;
            return {0, true};
        }

        auto newTokenId = llama_sampler_sample(sampler, context, -1);

        if (llama_token_is_eog(model, newTokenId)) {
            std::cout << "Ended due to EOG" << std::endl;
            return {newTokenId, true};
        }

        return {newTokenId, false};
    };

    //Re implement number of tokens to gen!!!

    auto [newTokenId, isEnd] = gen(firstBatch);
    if (isEnd) return "";

    int rewindPos = 0;
    int rewindTokenId = 0;
    int samplerChainPos = -1;
    while (!isEnd) {
        const auto piece = TokenToPiece(model, newTokenId).value();
        buffer += piece;

        if (!buffer.empty()) {
            MatchTrie::MatchResult matchResult;
            if (buffer.find_first_not_of(' ') != std::string::npos) {
                matchResult = matchingTrie.CheckBuffer(buffer.substr(buffer.find_first_not_of(' ')));
            } else {
                matchResult = matchingTrie.CheckBuffer(buffer);
            }
            if (matchResult == MatchTrie::MatchResult::NO) {
                WriteToReadbackBuffer(readbackBufferPtr, strdup(buffer.c_str()));
                response += buffer;
                buffer = "";
                //Rewind to last accepted id.
                rewindPos = llama_get_kv_cache_used_cells(context);
                rewindTokenId = newTokenId;

                if (samplerChainPos != -1) {
                    llama_sampler_chain_remove(sampler, samplerChainPos+2);
                    llama_sampler_chain_remove(sampler, samplerChainPos+1);
                    samplerChainPos = -1;
                }
            } else if (matchResult == MatchTrie::MatchResult::MATCHED_STOP) {
                //Matched a stop, break.
                break;
            } else if (matchResult == MatchTrie::MatchResult::MATCHED_REWIND) {
                llama_kv_cache_seq_rm(context, 0, rewindPos, llama_get_kv_cache_used_cells(context));

                const auto tokens = Tokenize(model, context, buffer, false, false);
                std::vector<llama_logit_bias> biases;
                biases.reserve(tokens.value().size());
                for (const llama_token token : tokens.value()) {
                    llama_logit_bias bias = {token, -50000.0f};
                    biases.push_back(bias);
                }

                samplerChainPos = llama_sampler_chain_n(sampler);
                sampler = LogitBiasSampler(sampler, model, static_cast<int32_t>(biases.size()), biases.data());
                sampler = DistSampler(sampler, 1337);

                buffer = "";
                newTokenId = rewindTokenId;
            }
        }

        const auto batch = llama_batch_get_one(&newTokenId, 1);
        std::tie(newTokenId, isEnd) = gen(batch);
    }

    PrintPerformanceInfo(context);
    readbackBufferPtr->done = true;
    return strdup(response.c_str());
}