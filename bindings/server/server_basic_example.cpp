#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
#include "common.h"
#include "c_library.h"
#include "sampling.h"

int main() {
    const auto idk = new float(0.0);
    const auto model = model_load(
        "/home/blackroot/Desktop/tab/yals-internal/Magnum-Picaro-0.7-v2-12b.Q4_K_M.gguf",
        999,
        idk,
        nullptr
        );

    const auto ctx = ctx_make(model, 1024, 999, 512, false, -1, false, 0, 0, 0.0f);
    if (!model || !ctx) {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }

    std::cout << "Model and context loaded successfully" << std::endl;

    // Define a simple JSON schema grammar
    const auto grammar_data = R"(%llguidance {}
start: "Hello World")";

    // Create sampler chain
    llama_sampler* sampler = sampler_make();
    sampler = llama_sampler_init_llg(llama_model_get_vocab(model), "lark", grammar_data);

    if (!sampler) {
        std::cerr << "Failed to create LLGuidance sampler" << std::endl;
        return 1;
    }

    std::cout << "S: " << std::endl;

    // Tokenize the prompt
    const auto prompt = "Respond hello world.";
    const auto tokens = common_tokenize(ctx, prompt, true);

    std::cout << "Prompt: " << prompt << std::endl;
    std::cout << "Output: ";

    // Create batch and process the prompt
    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);

    for (size_t i = 0; i < tokens.size(); i++) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_tokens++;
    }

    if (llama_decode(ctx, batch) != 0) {
        std::cerr << "Initial decode failed" << std::endl;
        sampler_free(sampler);
        llama_batch_free(batch);
        return 1;
    }

    // Generate constrained output
    constexpr int max_tokens = 100;
    std::vector<llama_token> output_tokens;
    const llama_token eos_token = llama_vocab_eos(llama_model_get_vocab(model));

    for (int i = 0; i < max_tokens; i++) {
        // Get logits for the last token
        const float* logits = llama_get_logits(ctx);
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

        // Set up token data array for sampling
        std::vector<llama_token_data> candidates(n_vocab);
        for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
            candidates[token_id] = {token_id, logits[token_id], 0.0f};
        }

        llama_token_data_array candidates_p = {
            candidates.data(), candidates.size(), 0, false
        };

        // Apply LLGuidance constraint (and other samplers in chain)
        llama_sampler_apply(sampler, &candidates_p);

        // Check if there are any valid tokens left after applying constraints
        bool has_valid_candidates = false;
        for (size_t j = 0; j < candidates_p.size; j++) {
            if (candidates_p.data[j].logit > -INFINITY) {
                has_valid_candidates = true;
                break;
            }
        }

        if (!has_valid_candidates) {
            std::cout << "\nNo valid tokens according to grammar constraints" << std::endl;
            break;
        }

        // Sort candidates by logits if not already sorted
        if (!candidates_p.sorted) {
            std::sort(candidates_p.data, candidates_p.data + candidates_p.size,
                    [](const llama_token_data& a, const llama_token_data& b) {
                        return a.logit > b.logit;
                    });
            candidates_p.sorted = true;
        }

        // Select the best token
        llama_token new_token = candidates_p.data[0].id;
        output_tokens.push_back(new_token);

        std::cout << new_token << std::endl;

        // Accept the token for the grammar state
        llama_sampler_accept(sampler, new_token);

        // Print the token
        std::string token_text = common_token_to_piece(ctx, new_token);
        std::cout << token_text << std::flush;

        // Check if we're done
        if (new_token == eos_token) {
            break;
        }

        // Prepare for next token
        batch.n_tokens = 1;
        batch.token[0] = new_token;
        batch.pos[0] = tokens.size() + i;

        // Process the next token
        if (llama_decode(ctx, batch) != 0) {
            std::cerr << "\nDecode failed" << std::endl;
            break;
        }
    }

    std::cout << "\nGeneration complete!" << std::endl;

    // Cleanup
    sampler_free(sampler);
    llama_batch_free(batch);

    return 0;
}
