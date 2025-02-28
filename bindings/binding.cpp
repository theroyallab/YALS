#include "binding.h"
#include "trie.hpp"
#include <iostream>
#include <optional>
#include <cstring>
#include <iomanip>
#include <llama-model.h>
#include <vector>
#include <string>
#include <sstream>

// Static vector to hold previous generation tokens
// TODO: Remove in continuous batch implementation
std::vector<llama_token> prevTokens;

void TestPrint(const char* text)
{
    std::cout << text << std::endl;
}

//typedef bool (*llama_progress_callback)(float progress, void * user_data);
llama_model* LoadModel(
    const char* modelPath,
    const int32_t numberGpuLayers,
    const float* tensorSplit,
    const llama_progress_callback callback)
{
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = numberGpuLayers;
    model_params.progress_callback = callback;

    model_params.split_mode = LLAMA_SPLIT_MODE_LAYER;
    model_params.tensor_split = tensorSplit;

    llama_model* model = llama_model_load_from_file(modelPath, model_params);

    return model;
}

char* GetModelChatTemplate(const llama_model* model) {
    static const char* tokenizerTemplateKey = "tokenizer.chat_template";
    const int32_t bufSize = llama_model_meta_val_str(model, tokenizerTemplateKey, nullptr, 0) + 1;

    const auto buffer = new char[bufSize];
    return buffer;
}

float GetModelFreqBase(const llama_model* model) {
    static const char* freqBaseKey = "general.rope_freq_base";

    // Get string length
    const int32_t bufSize = llama_model_meta_val_str(model, freqBaseKey, nullptr, 0) + 1;
    if (bufSize <= 1) {
        return 10000.0f; // Default if key not found
    }

    // Get string value
    std::vector<char> buffer(bufSize);
    const int32_t written = llama_model_meta_val_str(model, freqBaseKey, buffer.data(), bufSize);
    if (written <= 0) {
        return 10000.0f; // Default if read failed
    }

    try {
        // Convert string to float using string stream for locale-independent parsing
        std::stringstream ss(buffer.data());
        ss.imbue(std::locale::classic()); // Use classic locale for consistent decimal point
        float value;
        ss >> value;
        
        if (ss.fail()) {
            return 10000.0f; // Default on parse failure
        }
        
        return value;
    } catch (...) {
        return 10000.0f; // Default on any error
    }
}

llama_context *InitiateCtx(
    llama_model* model,
    const unsigned contextLength, // 0 = Use from model config
    const int32_t numberGpuLayers,
    const unsigned numBatches,
    const bool flashAttn,
    const float ropeFreqBase,
    const bool useYarn,
    const int kCacheQuantType,
    const int vCacheQuantType,
    const float kvDefragThreshold // -1 to disable
    )
{
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = contextLength;
    ctx_params.n_batch = numBatches;
    ctx_params.n_ubatch = numBatches;
    ctx_params.no_perf = false;
    ctx_params.flash_attn = flashAttn;

    ctx_params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_NONE;

    float freqBaseTrain = GetModelFreqBase(model);

    // Yarn, allegedly ext_factor -1 to default to model cfg but it looks sussy.
    // Only set linear RoPE if freq base is greater than the trained base
    if (useYarn) {
        ctx_params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_YARN;
        ctx_params.yarn_ext_factor = -1;
    } else if (ropeFreqBase > freqBaseTrain) {
        ctx_params.rope_scaling_type = LLAMA_ROPE_SCALING_TYPE_LINEAR;
        ctx_params.rope_freq_base = ropeFreqBase;
        ctx_params.rope_freq_scale = 0;
    }

    // Decrease CPU threads if model is fully offloaded on GPU
    if (numberGpuLayers >= llama_model_n_layer(model) || numberGpuLayers == -1) {
        ctx_params.n_threads = 1;
        ctx_params.n_threads_batch = 1;
    }

    ctx_params.type_k = static_cast<ggml_type>(kCacheQuantType);
    ctx_params.type_v = static_cast<ggml_type>(vCacheQuantType);
    ctx_params.defrag_thold = kvDefragThreshold;
    llama_context* ctx = llama_init_from_model(model, ctx_params);

    if (ctx == nullptr) {
        std::cerr << "error: couldn't make llama ctx in InitiateCtx()" << std::endl;
        return nullptr;
    }

    return ctx;
}

llama_token BosToken(const llama_model* model)
{
    return llama_vocab_bos(&model->vocab);
}

llama_token EosToken(const llama_model* model)
{
    return llama_vocab_eos(&model->vocab);
}

llama_token EotToken(const llama_model* model)
{
    return llama_vocab_eot(&model->vocab);
}

const char* TokenToString(const llama_model* model, const llama_token token) {
    return llama_vocab_get_text(&model->vocab, token);
}

uint32_t MaxSeqLen(const llama_context* ctx)
{
    return llama_n_ctx(ctx);
}

void FreeSampler(llama_sampler* sampler)
{
    llama_sampler_free(sampler);
}

void FreeCtx(llama_context* ctx)
{
    prevTokens.empty();
    llama_free(ctx);
}

void ClearContextKVCache(llama_context* ctx)
{
    prevTokens.empty();
    llama_kv_cache_clear(ctx);
}

void FreeModel(llama_model* model)
{
    llama_model_free(model);
}

void PrintPerformanceInfo(const llama_context* context) {
    const auto data = llama_perf_context(context);

    // Calculate tokens per second by dividing number of tokens by time in seconds
    // Convert milliseconds to seconds by dividing by 1000
    const double prompt_tok_per_sec = data.n_p_eval / (data.t_p_eval_ms / 1000.0);
    const double gen_tok_per_sec = data.n_eval / (data.t_eval_ms / 1000.0);

    std::cout << "\n\n" << std::fixed << std::setprecision(2)
              << "Prompt Processing: " << prompt_tok_per_sec << " tok/s, "
              << "Text Generation: " << gen_tok_per_sec << " tok/s" << "\n" << std::endl;
}

struct ReadbackBuffer
{
    unsigned lastReadbackIndex {0};
    bool done {false};
    char* jsonOutputBuffer = nullptr;

    std::vector<char*>* data = new std::vector<char*>();
    std::vector<llama_token>* ids = new std::vector<llama_token>();
};

void ResetReadbackBuffer(ReadbackBuffer* buffer) {
    buffer->done = false;
    buffer->lastReadbackIndex = 0;
    //Keep capacity, no resize.
    buffer->data->clear();

    delete[] buffer->jsonOutputBuffer;
    buffer->jsonOutputBuffer = nullptr;
}

bool IsReadbackBufferDone(const ReadbackBuffer* buffer)
{
    return buffer->done;
}

ReadbackBuffer* CreateReadbackBuffer()
{
    return new ReadbackBuffer {};
}

void WriteToReadbackBuffer(const ReadbackBuffer* buffer, char* stringData, const llama_token token)
{
    buffer->data->push_back(stringData);
    buffer->ids->push_back(token);
}

bool ReadbackNext(ReadbackBuffer *buffer, char** outChar, llama_token* outToken)
{
    if (buffer->lastReadbackIndex >= buffer->data->size())
    {
        return false;
    }

    *outChar = buffer->data->at(buffer->lastReadbackIndex);
    *outToken = buffer->ids->at(buffer->lastReadbackIndex);
    buffer->lastReadbackIndex++;
    return true;
}

char* ReadbackJsonStatus(const ReadbackBuffer* buffer) {
    return buffer->jsonOutputBuffer;
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
    llama_sampler_chain_add(sampler, llama_sampler_init_grammar(&model->vocab, grammar, root));
    return sampler;
}

llama_sampler* DrySampler(llama_sampler* sampler, const llama_model* model, const float multiplier,
                          const float base, const int32_t allowed_length, const int32_t penalty_last_n,
                          const char** sequence_breakers, const size_t n_breakers)
{
    llama_sampler_chain_add(
        sampler,
        llama_sampler_init_dry(
            &model->vocab, llama_model_n_ctx_train(model), multiplier, base, allowed_length,
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
    llama_sampler_chain_add(sampler, llama_sampler_init_infill(&model->vocab));
    return sampler;
}

// Typically applied early in the sampling chain
llama_sampler* LogitBiasSampler(
    llama_sampler* sampler, const llama_model* model, const int32_t nBias, const llama_logit_bias* logitBias)
{
    llama_sampler_chain_add(
        sampler,
        llama_sampler_init_logit_bias(
            llama_vocab_n_tokens(&model->vocab),
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
    const int nVocab = llama_vocab_n_tokens(&model->vocab);
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
llama_sampler* PenaltiesSampler(llama_sampler* sampler,
                                const int penaltyLastN, const float penaltyRepeat,
                                const float penaltyFreq, const float penaltyPresent)
{
    llama_sampler_chain_add(sampler, llama_sampler_init_penalties(
        penaltyLastN, penaltyRepeat, penaltyFreq, penaltyPresent
    ));
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

llama_sampler* TopNSigmaSampler(llama_sampler* sampler, const float nSigma)
{
    llama_sampler_chain_add(sampler, llama_sampler_init_top_n_sigma(nSigma));
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
    const int n = llama_token_to_piece(&llamaModel->vocab, id, buf, sizeof(buf), 0, true);
    if (n < 0) {
        std::cerr << "error: failed to convert token to piece in TokenToPiece()" << std::endl;
        return std::nullopt;
    }

    return std::string{buf, static_cast<size_t>(n)};
}

int32_t* EndpointTokenize(
    const llama_model* llamaModel,
    const char* prompt,
    const bool addSpecial,
    const bool parseSpecial) {

    const int32_t promptLength = strlen(prompt);
    const int n_prompt = -llama_tokenize(&llamaModel->vocab, prompt, promptLength,
                                   nullptr, 0, addSpecial, parseSpecial);
    auto tokenArray = new int32_t[n_prompt + 1];
    tokenArray[0] = n_prompt;

    if (llama_tokenize(&llamaModel->vocab, prompt, promptLength,
    tokenArray + 1, n_prompt + 1,
        addSpecial, parseSpecial) < 0) {
        std::cerr << "error: failed to tokenize the prompt in TokenizePrompt()" << std::endl;
        return nullptr;
    }

    return tokenArray;
}

void EndpointFreeTokens(const int32_t* tokens) {
    delete[] tokens;
}

char* EndpointDetokenize(
    const llama_model* llamaModel,
    int32_t* tokens,
    size_t numTokens,
    size_t maxTextSize,
    const bool addSpecial,
    const bool parseSpecial) {
    auto outText = new char[maxTextSize];
    llama_detokenize(&llamaModel->vocab, tokens, numTokens, outText, maxTextSize, addSpecial, parseSpecial);
    return outText;
}

void EndpointFreeString(const char* str) {
    delete[] str;
}

std::optional<std::vector<llama_token>> Tokenize(
    const llama_model* llamaModel, const llama_context* context, const std::string_view& prompt,
    const bool addSpecial, const bool parseSpecial) {

    const int n_prompt = -llama_tokenize(&llamaModel->vocab, prompt.data(), prompt.size(),
                                       nullptr, 0, addSpecial, parseSpecial);
    std::vector<llama_token> tokenizedPrompt(n_prompt);

    if (llama_tokenize(&llamaModel->vocab, prompt.data(), prompt.size(),
        tokenizedPrompt.data(), tokenizedPrompt.size(),
        addSpecial, parseSpecial) < 0) {
        std::cerr << "error: failed to tokenize the prompt in TokenizePrompt()" << std::endl;
        return std::nullopt;
    }

    return tokenizedPrompt;
}

// Escapes a string's special characters
std::string EscapeString(const std::string& input) {
    std::stringstream ss;
    for (unsigned char ch : input) {
        switch (ch) {
            case '"':  ss << "\\\""; break;
            case '\\': ss << "\\\\"; break;
            case '\b': ss << "\\b";  break;
            case '\f': ss << "\\f";  break;
            case '\n': ss << "\\n";  break;
            case '\r': ss << "\\r";  break;
            case '\t': ss << "\\t";  break;
            default:
                if (ch < 0x20) {
                    // Escape ASCII control characters with \u00x
                    ss << "\\u" << std::setw(4) << std::setfill('0') << std::hex << static_cast<int>(ch);
                } else {
                    ss << ch;
                }
        }
    }

    return ss.str();
}

std::string MakeJsonOutputString(const llama_context* context, const std::string &finishReason, const std::string &stopToken) {
    const auto [t_start_ms, t_load_ms, t_p_eval_ms, t_eval_ms, n_p_eval, n_eval] = llama_perf_context(context);

    const double t_p_eval_s = t_p_eval_ms / 1000.0;
    const double t_eval_s = t_eval_ms / 1000.0;

    // Calculate tokens per second, return 0 if division by 0
    const double prompt_tokens_per_sec = t_p_eval_s > 0 ? n_p_eval / t_p_eval_s : 0;
    const double gen_tokens_per_sec = t_eval_s > 0 ? n_eval / t_eval_s : 0;

    // Escape the stop token to avoid errors in Deno
    const std::string escapedStopToken = EscapeString(stopToken);

    std::stringstream ss;
    ss << "{"
       << R"("promptTokens":)" << n_p_eval << ","
       << R"("genTokens":)" << n_eval << ","
       << R"("promptSec":)" << t_p_eval_s << ","
       << R"("genSec":)" << t_eval_s << ","
       << R"("genTokensPerSec":)" << gen_tokens_per_sec << ","
       << R"("promptTokensPerSec":)" << prompt_tokens_per_sec << ","
       << R"("finishReason": ")" << finishReason << "\","
       << R"("stopToken": ")" << escapedStopToken << "\""
       << "}";

    return ss.str();
}

// From llama.cpp/common/common.cpp
// Returns the point when prompt prefix starts to diverge
size_t common_lcp(const std::vector<llama_token> &a, const std::vector<llama_token> &b) {
    size_t i;
    for (i = 0; i < a.size() && i < b.size() && a[i] == b[i]; i++) {}

    return i;
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
    ggml_abort_callback abortCallback,
    const char** rewindStrings,
    const unsigned numRewindStrings,
    const char** stoppingStrings,
    const unsigned numStoppingStrings)
{
    if (abortCallback != nullptr) {
        llama_set_abort_callback(context, abortCallback, nullptr);
    }

    llama_perf_context_reset(context);

    std::string finishReason = "Unspecified";
    std::string stoppedAt;

    // Lambda function to process the prompt in chunked batches
    auto processPromptBatches = [&](std::vector<llama_token>& tokens) -> bool {
        const int batchSize = llama_n_batch(context);
        int n_ctx = llama_n_ctx(context);

        if (tokens.size() > n_ctx) {
            finishReason = "CtxExceeded";
            return false;
        }

        // Check when tokens diverge and remove everything after the common prefix
        const size_t prefixEnd = common_lcp(tokens, prevTokens);
        llama_kv_cache_seq_rm(context, 0, prefixEnd, -1);

        for (size_t i = prefixEnd; i < tokens.size(); i += batchSize) {
            const size_t remaining = tokens.size() - i;
            const size_t currentBatchSize = std::min(remaining, static_cast<size_t>(batchSize));

            const llama_batch batch = llama_batch_get_one(
                tokens.data() + i,
                currentBatchSize
            );

            if (llama_decode(context, batch)) {
                finishReason = "BatchDecode";
                return false;
            }
        }

        prevTokens = tokens;
        return true;
    };

    // Tokenize and determine the amount of tokens to generate
    auto promptTokens = Tokenize(model, context, prompt, addSpecial, parseSpecial).value();

    // Process the prompt in chunked batches
    if (!processPromptBatches(promptTokens)) {
        readbackBufferPtr->jsonOutputBuffer = strdup(MakeJsonOutputString(context, finishReason, stoppedAt).c_str());
        readbackBufferPtr->done = true;
        return nullptr;
    }

    MatchTrie::MatchTrie matchingTrie;
    matchingTrie.AddMatchableWords(rewindStrings, numRewindStrings, MatchTrie::MatchType::REWIND);
    matchingTrie.AddMatchableWords(stoppingStrings, numStoppingStrings, MatchTrie::MatchType::STOP);

    std::string response;
    std::string buffer;

    // Kickstart generation with a single token
    llama_token firstToken = promptTokens.back();
    llama_batch firstBatch = llama_batch_get_one(&firstToken, 1);

    // Continue generation
    auto gen = [&](const llama_batch& batch, llama_sampler* smpl) -> std::pair<llama_token, bool> {
        if (llama_decode(context, batch)) {
            finishReason = "BatchDecode";
            return {0, true};
        }

        llama_token newTokenId = llama_sampler_sample(smpl, context, -1);

        if (llama_vocab_is_eog(&model->vocab, newTokenId)) {
            return {newTokenId, true};
        }

        return {newTokenId, false};
    };

    // Append to the generation batch
    auto [newTokenId, isEnd] = gen(firstBatch, sampler);

    // Extra samplers - Banned strings
    int rewindPos = 0;
    int rewindTokenId = 0;
    int tokenCount = 0;
    int rewindTokenCount = 0;
    std::vector<llama_logit_bias> biases;
    llama_sampler* banSampler = nullptr;
    llama_batch batch = firstBatch;
    while (true) {
        // Abort if callback is fired
        if (isEnd) {
            finishReason = "StopToken";
            stoppedAt = TokenToPiece(model, newTokenId).value();
            break;
        }

        // End on length if max tokens is exceeded
        if (tokenCount + batch.n_tokens > numberTokensToPredict) {
            finishReason = "MaxNewTokens";
            stoppedAt = TokenToPiece(model, newTokenId).value();
            break;
        }

        const auto piece = TokenToPiece(model, newTokenId).value();

        buffer += piece;
        tokenCount += batch.n_tokens;

        if (!buffer.empty()) {
            MatchTrie::MatchResult matchResult;
            size_t nonSpacePos = buffer.find_first_not_of(' ');

            // Strip leading spaces which is a common issue that will confuse the trie matching.
            if (nonSpacePos != std::string::npos) {
                matchResult = matchingTrie.CheckBuffer(buffer.substr(nonSpacePos));
            } else {
                matchResult = matchingTrie.CheckBuffer(buffer);
            }

            if (matchResult == MatchTrie::MatchResult::NO) {
                WriteToReadbackBuffer(readbackBufferPtr, strdup(buffer.c_str()), newTokenId);
                response += buffer;
                buffer = "";

                // Save last known accept point in case we have to rewind back to the last accept.
                rewindPos = llama_get_kv_cache_used_cells(context);
                rewindTokenId = newTokenId;
                rewindTokenCount = tokenCount;

                // If we had a rewind state built, tear it down as we've accepted a sequence.
                if (banSampler != nullptr) {
                    llama_sampler_free(banSampler);
                    banSampler = nullptr;
                    biases.clear();
                }
            } else if (matchResult == MatchTrie::MatchResult::MATCHED_STOP) {
                // Matched a stop, break.
                finishReason = "StopString";
                stoppedAt = TokenToPiece(model, newTokenId).value();
                break;
            } else if (matchResult == MatchTrie::MatchResult::MATCHED_REWIND) {
                llama_kv_cache_seq_rm(context, 0, rewindPos, -1);

                const auto tokens = Tokenize(model, context, buffer, false, false);
                for (const llama_token token : tokens.value()) {
                    biases.push_back({token, -50000.0f});
                }

                if (banSampler == nullptr) {
                    banSampler = MakeSampler();
                    LogitBiasSampler(banSampler, model, static_cast<int32_t>(biases.size()), biases.data());
                    DistSampler(banSampler, 1337);
                } else {
                    llama_sampler_chain_remove(banSampler, 1);
                    llama_sampler_chain_remove(banSampler, 0);
                    LogitBiasSampler(banSampler, model, static_cast<int32_t>(biases.size()), biases.data());
                    DistSampler(banSampler, 1337);
                }

                buffer = "";
                newTokenId = rewindTokenId;

                batch = llama_batch_get_one(&newTokenId, 1);
                std::tie(newTokenId, isEnd) = gen(batch, banSampler);
                tokenCount = rewindTokenCount;
                continue;
            }
        }

        batch = llama_batch_get_one(&newTokenId, 1);
        std::tie(newTokenId, isEnd) = gen(batch, sampler);
    }

    if (banSampler != nullptr) {
        llama_sampler_free(banSampler);
    }

    readbackBufferPtr->jsonOutputBuffer = strdup(MakeJsonOutputString(context, finishReason, stoppedAt).c_str());
    readbackBufferPtr->done = true;
    return strdup(response.c_str());
}