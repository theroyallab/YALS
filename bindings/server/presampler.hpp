#ifndef PRESAMPLER_HPP
#define PRESAMPLER_HPP

#include <unordered_set>
#include "samplers.hpp"

/*
 * The presampler is responsible for rewind biasing and stopping biasing.
 *
 *  Provides:
 *  Minimum token generation. (By banning the stop token)
 *  Rewind bans: Keeps track of the rewinding ban buffer.
 *
 *  Mechanism:
 *  This is overall simply an extra sampler that is used first in the sampling chain to pre-filter banned logits.
 */

inline llama_sampler* build_presampler_chain(
    const llama_model* model,
    const uint32_t seed,
    const int32_t n_bias,
    const llama_logit_bias* logit_bias) {
    llama_sampler* sampler = sampler_make();
    sampler = sampler_logit_bias(sampler, model, n_bias, logit_bias);
    sampler = sampler_dist(sampler, seed);

    return sampler;
}

struct Presampler {
private:
    //Biases imposed by the rewind mechanism.
    std::unordered_set<llama_token> rewind_biases;

    //Biases imposed by stopping criterion.
    std::unordered_set<llama_token> eos_biases;

    void rebuild_presampler(const llama_model* model) {
        std::vector<llama_logit_bias> biases;
        for (const llama_token token : rewind_biases) {
            biases.push_back({token, -50000.0f});
        }
        for (const llama_token token : eos_biases) {
            biases.push_back({token, -50000.0f});
        }

        should_presample = !biases.empty();

        llama_sampler_free(sampler);
        sampler = build_presampler_chain(model, seed, static_cast<int32_t>(biases.size()), biases.data());
    }

    void add_tokens_to_bias(std::unordered_set<llama_token>& bias_set,
                            const llama_model* model,
                            const std::vector<llama_token>& tokens) {
        for (auto& token : tokens) {
            bias_set.insert(token);
        }
        rebuild_presampler(model);
    }

public:
    llama_sampler* sampler {nullptr};
    uint32_t seed = 1337;
    bool should_presample = false;

    void add_rewind_bans(const llama_model* model, const std::vector<llama_token> &tokens) {
        add_tokens_to_bias(rewind_biases, model, tokens);
    }

    void add_eos_ban(const llama_model* model, const std::vector<llama_token> &tokens) {
        add_tokens_to_bias(eos_biases, model, tokens);
    }

    void clear_rewind_bans(const llama_model* model) {
        if (rewind_biases.empty()) {
            return;
        }
        rewind_biases.clear();
        rebuild_presampler(model);
    }

    void clear_eos_bans(const llama_model* model) {
        if (eos_biases.empty()) {
            return;
        }
        eos_biases.clear();
        rebuild_presampler(model);
    }

    // Fully resets the presampler state
    void reset() {
        rewind_biases.clear();
        eos_biases.clear();

        // Might not be needed, but just in case
        should_presample = false;

        if (sampler) {
            llama_sampler_free(sampler);
            sampler = nullptr;
        }
    }
};

#endif //PRESAMPLER_HPP
