#ifndef SAMPLERS_HPP
#define SAMPLERS_HPP

#include <llama-model.h>
#include "sampling.h"

/*
 * A very minimal abstraction over lcpp samplers primarily to expose to bindings.
 */

llama_sampler* sampler_make() {
    llama_sampler_chain_params params = llama_sampler_chain_default_params();
    params.no_perf = false;
    return llama_sampler_chain_init(params);
}

template<typename T>
llama_sampler* add_sampler(llama_sampler* chain, T* sampler) {
    llama_sampler_chain_add(chain, sampler);
    return chain;
}

void sampler_free(llama_sampler* sampler) {
    llama_sampler_free(sampler);
}

llama_sampler* sampler_llguidance(llama_sampler* chain, const llama_model* model, const char* grammar_kind, const char* grammar_data) {
    return add_sampler(chain, llama_sampler_init_llg(llama_model_get_vocab(model), grammar_kind, grammar_data));
}

llama_sampler* sampler_dist(llama_sampler* chain, const uint32_t seed) {
    return add_sampler(chain, llama_sampler_init_dist(seed));
}

llama_sampler* sampler_greedy(llama_sampler* chain) {
    return add_sampler(chain, llama_sampler_init_greedy());
}

llama_sampler* sampler_min_p(llama_sampler* chain, const float min_p, const size_t min_keep) {
    return add_sampler(chain, llama_sampler_init_min_p(min_p, min_keep));
}

llama_sampler* sampler_mirostat_v2(llama_sampler* chain, const uint32_t seed, const float tau, const float eta) {
    return add_sampler(chain, llama_sampler_init_mirostat_v2(seed, tau, eta));
}

llama_sampler* sampler_penalties(llama_sampler* chain, const int penalty_last_n, const float penalty_repeat,
                                 const float penalty_freq, const float penalty_present) {
    return add_sampler(chain, llama_sampler_init_penalties(
        penalty_last_n, penalty_repeat, penalty_freq, penalty_present));
}

llama_sampler* sampler_temp(llama_sampler* chain, const float temp) {
    return add_sampler(chain, llama_sampler_init_temp(temp));
}

llama_sampler* sampler_temp_ext(llama_sampler* chain, const float temp,
                                const float dynatemp_range, const float dynatemp_exponent) {
    return add_sampler(chain, llama_sampler_init_temp_ext(temp, dynatemp_range, dynatemp_exponent));
}

llama_sampler* sampler_top_k(llama_sampler* chain, const int top_k) {
    return add_sampler(chain, llama_sampler_init_top_k(top_k));
}

llama_sampler* sampler_top_p(llama_sampler* chain, const float top_p, const size_t min_keep) {
    return add_sampler(chain, llama_sampler_init_top_p(top_p, min_keep));
}

llama_sampler* sampler_typical(llama_sampler* chain, const float typical_p, const size_t min_keep) {
    return add_sampler(chain, llama_sampler_init_typical(typical_p, min_keep));
}

llama_sampler* sampler_top_n_sigma(llama_sampler* chain, const float n_sigma) {
    return add_sampler(chain, llama_sampler_init_top_n_sigma(n_sigma));
}

llama_sampler* sampler_xtc(llama_sampler* chain, const float xtc_probability, const float xtc_threshold,
                           const size_t min_keep, const uint32_t seed) {
    return add_sampler(chain, llama_sampler_init_xtc(xtc_probability, xtc_threshold, min_keep, seed));
}

llama_sampler* sampler_grammar(llama_sampler* chain, const llama_model* model,
                               const char* grammar, const char* root) {
    return add_sampler(chain, llama_sampler_init_grammar(&model->vocab, grammar, root));
}

llama_sampler* sampler_dry(llama_sampler* chain, const llama_model* model, const float multiplier,
                           const float base, const int32_t allowed_length, const int32_t penalty_last_n,
                           const char** sequence_breakers, const size_t n_breakers) {
    return add_sampler(chain, llama_sampler_init_dry(
        &model->vocab, llama_model_n_ctx_train(model), multiplier, base, allowed_length,
        penalty_last_n, sequence_breakers, n_breakers));
}

llama_sampler* sampler_infill(llama_sampler* chain, const llama_model* model) {
    return add_sampler(chain, llama_sampler_init_infill(&model->vocab));
}

llama_sampler* sampler_logit_bias(llama_sampler* chain, const llama_model* model,
                                  const int32_t n_bias, const llama_logit_bias* logit_bias) {
    return add_sampler(chain, llama_sampler_init_logit_bias(
        llama_vocab_n_tokens(&model->vocab), n_bias, logit_bias));
}

llama_sampler* sampler_mirostat(llama_sampler* chain, const llama_model* model, const uint32_t seed,
                                const float tau, const float eta, const int m) {
    const int n_vocab = llama_vocab_n_tokens(&model->vocab);
    return add_sampler(chain, llama_sampler_init_mirostat(n_vocab, seed, tau, eta, m));
}

#endif // SAMPLERS_HPP