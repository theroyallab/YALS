#ifndef MULTISAMPLER_HPP
#define MULTISAMPLER_HPP

#include <cmath>
#include "llama.h"
#include "presampler.hpp"
#include "sampling.h"
#include <optional>

/*
 * Multisampler, since sampler chain doesn't work for grammar sampling.
 * It's otherwise just basically doing the same thing as sampler chain.
 */

struct MultistageSampler {
    llama_sampler* constraint_sampler = nullptr;
    Presampler presampler;
    llama_sampler* sampler = nullptr;
    const llama_vocab* vocab{};

    std::vector<llama_token_data> candidates;

    explicit MultistageSampler(const llama_model* model) :
        presampler(), vocab(llama_model_get_vocab(model)) {
        candidates.resize(llama_vocab_n_tokens(vocab));
    }

    void constrain(const char* grammar_data) {
        static auto grammar_kind = "lark";
        constraint_sampler = llama_sampler_init_llg(vocab, grammar_kind, grammar_data);
    }

    std::optional<llama_token> sample(llama_context* ctx, const int index) {
        const float* logits = llama_get_logits_ith(ctx, index);

        if (candidates.size() < llama_vocab_n_tokens(vocab)) {
            candidates.resize(llama_vocab_n_tokens(vocab));
        }

        for (llama_token token_id = 0; token_id < llama_vocab_n_tokens(vocab); token_id++) {
            candidates[token_id] = {token_id, logits[token_id], 0.0f};
        }

        llama_token_data_array candidates_array = {
            candidates.data(),
            candidates.size(),
            -1,
            false
        };

        // Apply constraints first as they're mandatory
        if (constraint_sampler != nullptr) {
            llama_sampler_apply(constraint_sampler, &candidates_array);
        }

        // Apply the presampler next, these will narrow down the possible output logits.
        if (presampler.should_presample) {
            llama_sampler_apply(presampler.sampler, &candidates_array);
        }

        bool has_valid_tokens = false;
        for (size_t i = 0; i < candidates_array.size; i++) {
            if (candidates_array.data[i].logit != -INFINITY) {
                has_valid_tokens = true;
                break;
            }
        }

        auto accept_all = [this](const llama_token token) {
            if (constraint_sampler != nullptr) {
                llama_sampler_accept(constraint_sampler, token);
            }

            if (presampler.should_presample) {
                llama_sampler_accept(presampler.sampler, token);
            }

            llama_sampler_accept(sampler, token);
        };

        if (!has_valid_tokens) {
            const llama_token eos_token = llama_vocab_eot(vocab);
            accept_all(eos_token);
            return std::nullopt;
        }

        llama_sampler_apply(sampler, &candidates_array);
        llama_token selected_token = candidates_array.data[candidates_array.selected].id;

        accept_all(selected_token);

        return selected_token;
    }
};

#endif //MULTISAMPLER_HPP