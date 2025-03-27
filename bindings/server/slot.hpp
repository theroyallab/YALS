#ifndef SLOT_HPP
#define SLOT_HPP

#include <string>
#include <vector>
#include "llama.h"
#include "multisampler.hpp"
#include "tokenization.hpp"
#include "sequence_stream.hpp"
#include "presampler.hpp"
#include "readback_buffer.hpp"

/*
 *  Slots are essentially just a data container holding the current inference state for a single complete inference.
 *
 *  Provides
 *  A centralized data container for the processor to manage the inference state.
 */

struct Slot {
    enum class State {
        IDLE,
        PROMPT,
        GENERATING
    };

    struct SlotSnapshot {
        size_t prompt_tokens_processed{};
        int tokens_generated{};
        int n_past{};
        int i_batch{};
        llama_token last_token{};
        std::string previous_seq_stream_buffer;
        int32_t previous_kv_pos{};
        bool last_token_prompt{};

        static SlotSnapshot snapshot_slot(const Slot& slot, llama_context* ctx, const bool during_prompt) {
            SlotSnapshot snapshot;
            snapshot.prompt_tokens_processed = slot.prompt_tokens_processed;
            snapshot.tokens_generated = slot.tokens_generated;
            snapshot.n_past = slot.n_past;
            snapshot.i_batch = slot.i_batch;
            snapshot.last_token = slot.last_token;
            snapshot.previous_seq_stream_buffer = slot.sequence_stream->sequence_buffer;

            snapshot.last_token_prompt = during_prompt;

            // During the prompt because we do not call decode, we need a special case to update the kv pos for prompt
            snapshot.previous_kv_pos = during_prompt ? slot.n_past : llama_kv_self_seq_pos_max(ctx, slot.slot_id);
            return snapshot;
        }

        int32_t rewind_slot(Slot& slot) const {
            slot.prompt_tokens_processed = prompt_tokens_processed;
            slot.tokens_generated = tokens_generated;
            slot.n_past = n_past;
            slot.i_batch = i_batch;
            slot.last_token = last_token;
            slot.sequence_stream->sequence_buffer = previous_seq_stream_buffer;
            return previous_kv_pos;
        }
    };

    int job_index = -1;
    int request_id = -1;
    int slot_id = 0;
    State state = State::IDLE;

    std::vector<llama_token> prompt_tokens;
    size_t prompt_tokens_processed = 0;
    int tokens_generated = 0;

    int n_past = 0;
    int i_batch = -1;

    bool test_safeguard = false;

    llama_token last_token = 0;
    std::string generated_text;

    TokenStreamDetokenizer* detokenizer;
    SequenceStream* sequence_stream;
    MultistageSampler multi_sampler;
    InferenceArgs inference_args;
    SlotSnapshot rewind_snapshot;
    ReadbackBuffer* readback_buffer = nullptr;

    explicit Slot(const llama_model* model, llama_context* ctx): multi_sampler(model) {
        detokenizer = new TokenStreamDetokenizer(ctx);
        sequence_stream = new SequenceStream();
    }

    ~Slot() {
        delete detokenizer;
        delete sequence_stream;
    }

    [[nodiscard]] bool is_processing() const { return state != State::IDLE; }
    [[nodiscard]] bool is_processing_prompt() const { return state == State::PROMPT; }
    [[nodiscard]] bool is_generating() const { return state == State::GENERATING; }

    void clear() {
        request_id = -1;
        state = State::IDLE;
        prompt_tokens_processed = 0;
        tokens_generated = 0;
        n_past = 0;
        i_batch = -1;
        last_token = 0;
        generated_text.clear();
        detokenizer->reset();
    }

    void print_dbg_info(llama_context* ctx) const {
        std::cout << "=== Slot Debug Info ===\n";
        std::cout << "KV cache size: " << llama_kv_self_seq_pos_max(ctx, slot_id) << std:: endl;
        std::cout << "job_index: " << job_index << "\n";
        std::cout << "request_id: " << request_id << "\n";
        std::cout << "slot_id: " << slot_id << "\n";

        std::cout << "state: ";
        switch (state) {
            case State::IDLE:      std::cout << "IDLE"; break;
            case State::PROMPT:    std::cout << "PROMPT"; break;
            case State::GENERATING: std::cout << "GENERATING"; break;
            default:               std::cout << static_cast<int>(state); break;
        }
        std::cout << "\n";

        std::cout << "prompt_tokens.size(): " << prompt_tokens.size() << "\n";
        std::cout << "prompt_tokens_processed: " << prompt_tokens_processed << "\n";
        std::cout << "tokens_generated: " << tokens_generated << "\n";
        std::cout << "n_past: " << n_past << "\n";
        std::cout << "i_batch: " << i_batch << "\n";
        std::cout << "last_token: " << last_token << "\n";
        std::cout << "generated_text: \"" << generated_text << "\"\n";
        std::cout << "======================\n";
    }

    void end(const int new_id, llama_context* ctx) {
        clear();
        job_index = new_id;
    }
};

#endif // SLOT_HPP