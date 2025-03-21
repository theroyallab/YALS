#ifndef PROCESSOR_HPP
#define PROCESSOR_HPP

#include <utility>
#include <vector>
#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <cmath>
#include <iostream>
#include <thread>

#include "inference_args.hpp"
#include "llama.h"
#include "tokenization.hpp"
#include "slot.hpp"
#include "request.hpp"
#include "sequence_stream.hpp"

/*
 * Primary server processor. Controls the overall flow. This processes in slot-order and does not
 * guarantee fairness in processing, to avoid overly shuffling the kv-cache.
 *
 * Provides:
 * The primary job-submit interface
 * Continuous batching aka High-efficiency Multi-user inference
 * Slot state management (Idle, Processing Prompt, Generating)
 * Slot Rewinding
 * Runs the actual llama model forward
 * Job cancellation
 *
 * Mechanism:
 * It's a server.
 */

class Processor {
    llama_model* model;
    llama_context* ctx;
    llama_batch batch{};
    bool abort_inference = false;

    std::vector<Slot> slots;
    uint32_t batch_size;

    std::queue<Request> queue_tasks;
    std::mutex mutex_tasks;
    std::condition_variable cv_tasks;

    std::thread worker_thread;
    std::atomic<bool> should_exit{false};

    int current_job_index = 0;
    Tokenizer tokenizer;

    // nearly eq to common_add_to_batch from lcpp server
    void add_to_batch(Slot& slot, const llama_token token, const bool compute_logits) {
        slot.i_batch = batch.n_tokens;

        batch.token[batch.n_tokens] = token;
        batch.pos[batch.n_tokens] = slot.n_past;
        batch.n_seq_id[batch.n_tokens] = 1;
        batch.seq_id[batch.n_tokens][0] = slot.job_index;
        batch.logits[batch.n_tokens] = static_cast<int8_t>(compute_logits);

        batch.n_tokens++;
        slot.n_past++;
    }

    // Derived from lcpp server originally
    static llama_pos common_longest_prefix(const std::vector<llama_token>& a, const std::vector<llama_token>& b) {
        llama_pos i;
        for (i = 0; i < a.size() && i < b.size() && a[i] == b[i]; i++) {}
        return i;
    }

    //Tasks are not processed in fairness.
    //A task assigned to a slot sticks to it until finished to avoid shuffling the cache.
    //This is not a fair processing scheme, however it is more optimal
    void process_tasks() {
        bool has_idle_slot = false;
        for (const auto& slot : slots) {
            if (slot.state == Slot::State::IDLE) {
                has_idle_slot = true;
                break;
            }
        }

        if (!has_idle_slot) {
            return;
        }

        std::unique_lock lock(mutex_tasks);
        if (queue_tasks.empty()) {
            return;
        }

        const auto [id,
            prompt_tokens,
            inference_args,
            readback_buffer] = queue_tasks.front();

        queue_tasks.pop();
        lock.unlock();

        //Check for the best slot. The best slot is the one with the longest prefix.
        Slot* best_slot = nullptr;
        llama_pos longest_prefix = 0;
        Slot* oldest_idle_slot = nullptr;
        for (auto& slot : slots) {
            if (slot.state == Slot::State::IDLE) {
                if (!oldest_idle_slot || slot.job_index < oldest_idle_slot->job_index) {
                    oldest_idle_slot = &slot;
                }

                const llama_pos prefix_len = common_longest_prefix(prompt_tokens, slot.prompt_tokens);
                const bool is_better = prefix_len > longest_prefix ||
                                      (prefix_len == longest_prefix &&
                                       (!best_slot || slot.job_index < best_slot->job_index));

                if (is_better) {
                    longest_prefix = prefix_len;
                    best_slot = &slot;
                }
            }
        }
        //If we do not have any prefix matches, pick the oldest idle slot.
        if (longest_prefix == 0) {
            best_slot = oldest_idle_slot;
        }

        if (!best_slot)
            return;

        if (longest_prefix > 0) {
            // Reuse prefix, cut the KV to the prefix size and adjust to gen or prompt appropriately.
            llama_kv_self_seq_rm(ctx, best_slot->job_index, longest_prefix, -1);
            best_slot->prompt_tokens_processed = longest_prefix;
            best_slot->tokens_generated = 0;
            best_slot->generated_text.clear();

            best_slot->n_past = llama_kv_self_seq_pos_max(ctx, best_slot->job_index);

            best_slot->state =
                longest_prefix == prompt_tokens.size() ?
                Slot::State::GENERATING :
                Slot::State::PROMPT;
        } else {
            // Nothing to reuse, clear the kv and start fresh
            best_slot->clear(ctx);
            best_slot->prompt_tokens_processed = 0;
            best_slot->state = Slot::State::PROMPT;
        }

        best_slot->request_id = id;
        best_slot->prompt_tokens = prompt_tokens;
        best_slot->inference_args = inference_args;
        best_slot->readback_buffer = readback_buffer;

        best_slot->sequence_stream->bind_sequences(inference_args.stopping_strings, inference_args.rewind_strings);
        best_slot->rewind_snapshot = Slot::SlotSnapshot::snapshot_slot(*best_slot, ctx);

        // Ban the EOS tokens immediately before starting generation if we have min tokens.
        if (inference_args.min_tokens_to_gen > 0 && inference_args.min_tokens_to_gen < inference_args.max_tokens_to_gen) {
            const std::vector terminal_token_bans {llama_vocab_eos(llama_model_get_vocab(model)), llama_vocab_eot(llama_model_get_vocab(model))};
            best_slot->presampler.add_eos_ban(model, terminal_token_bans);
        }
    }

    void update_prompt_slots() {
        for (auto& slot : slots) {
            if (slot.is_processing_prompt() && slot.prompt_tokens_processed < slot.prompt_tokens.size()) {
                while (slot.prompt_tokens_processed < slot.prompt_tokens.size() &&
                       batch.n_tokens < batch_size) {

                    const llama_token token = slot.prompt_tokens[slot.prompt_tokens_processed];
                    const bool is_last_prompt_token = (slot.prompt_tokens_processed == slot.prompt_tokens.size() - 1);

                    slot.i_batch = batch.n_tokens;

                    add_to_batch(slot, token, is_last_prompt_token);
                    slot.prompt_tokens_processed++;
                }

                if (slot.prompt_tokens_processed >= slot.prompt_tokens.size()) {
                    slot.state = Slot::State::GENERATING;
                }
            }
        }
    }

    // Processes the next sequence token. Finalizes the request if gen is finished.
    bool process_token(Slot& slot, const llama_token token) const {
        const auto piece_opt = slot.detokenizer->process_token(token, true);
        if (!piece_opt) {
            std::cerr << "error: failed to process token " << token << std::endl;
            return false;
        }

        slot.tokens_generated++;

        const bool is_eos = tokenizer.is_eos_token(token);
        bool is_complete = is_eos || slot.tokens_generated >= slot.inference_args.max_tokens_to_gen;
        bool yield_final = false;
        const std::string& piece = piece_opt.value_or("");

        if (!piece.empty()) {
            std::string out_string;
            const auto result = slot.sequence_stream->append(piece, token, out_string);

            switch (result) {
                case SequenceStream::Continuation::ACCEPT:
                    slot.generated_text += out_string;
                    slot.presampler.clear_rewind_bans(model);
                    slot.rewind_snapshot = Slot::SlotSnapshot::snapshot_slot(slot, ctx);
                    yield_final = true;

                    if (slot.inference_args.min_tokens_to_gen > 0
                        && slot.tokens_generated >= slot.inference_args.min_tokens_to_gen
                        && slot.inference_args.min_tokens_to_gen < slot.inference_args.max_tokens_to_gen) {
                        slot.presampler.clear_eos_bans(model);
                    }
                    break;
                case SequenceStream::Continuation::REWIND: {
                    //Restore the slot to whatever the last accepted snapshot was.
                    //Then delete the part of the KV we're rewinding
                    const int32_t prev_kv_pos = slot.rewind_snapshot.rewind_slot(slot);
                    llama_kv_self_seq_rm(ctx, slot.job_index, prev_kv_pos, -1);

                    //Ban every token in the buffer.
                    const auto tokens = tokenizer.tokenize(out_string, false, false);
                    if (!tokens) {
                        //Todo::@Z This is a hard error as it will infinite loop if tokens is empty here.
                    }
                    slot.presampler.add_rewind_bans(model, tokens.value());

                    //It's possible we rewind to before the min token threshold, so we need to ensure the eos tokens are actually banned.
                    if (slot.inference_args.min_tokens_to_gen > 0 && slot.inference_args.min_tokens_to_gen < slot.inference_args.max_tokens_to_gen) {
                        const std::vector terminal_token_bans {llama_vocab_eos(llama_model_get_vocab(model)), llama_vocab_eot(llama_model_get_vocab(model))};
                        slot.presampler.add_eos_ban(model, terminal_token_bans);
                    }
                    }
                    return true;
                case SequenceStream::Continuation::STOP:
                    is_complete = true;
                    break;
                case SequenceStream::Continuation::BUFFER:
                    break;
            }
        }

        if (!is_complete) {
            if (!piece.empty()) {
                readback_write_to_buffer(slot.readback_buffer, piece, token);
            }
            return !is_eos;
        }

        std::string final_piece = piece;
        if (slot.detokenizer->has_incomplete()) {
            const std::string remaining = slot.detokenizer->flush();
            if (!remaining.empty()) final_piece += remaining;
        }

        if (yield_final && !final_piece.empty()) {
            readback_write_to_buffer(slot.readback_buffer, final_piece, token);
        }

        //TODO::@Z JSON status.
        readback_finish(slot.readback_buffer, R"({"status": "Job Completed", "reason": "Normal completion"})");
        return false;
    }

    void update_gen_slots() {
        for (auto& slot : slots) {
            if (slot.is_generating() && batch.n_tokens < batch_size) {
                add_to_batch(slot, slot.last_token, true);
            }
        }

        if (batch.n_tokens == 0) {
            return;
        }

        if (llama_decode(ctx, batch) != 0) {
            //TODO @Z:: Log error.
            return;
        }

        for (auto& slot : slots) {
            if (slot.i_batch < 0 || slot.i_batch >= batch.n_tokens) {
                continue;
            }

            if (slot.is_generating()) {
                llama_token token;

                // If we have a presampler, we append our main sampler to it, otherwise we just use our main sampler.
                if (slot.presampler.should_presample) {
                    const int presampler_tail = llama_sampler_chain_n(slot.presampler.sampler);
                    llama_sampler_chain_add(slot.presampler.sampler, slot.inference_args.sampler);
                    token = llama_sampler_sample(slot.presampler.sampler, ctx, slot.i_batch);
                    llama_sampler_chain_remove(slot.presampler.sampler, presampler_tail);
                } else {
                    token = llama_sampler_sample(slot.inference_args.sampler, ctx, slot.i_batch);
                }
                llama_sampler_accept(slot.inference_args.sampler, token);
                slot.last_token = token;
                slot.i_batch = -1;

                if (const bool continue_gen = process_token(slot, token); !continue_gen) {
                    slot.end(++current_job_index, ctx);
                    //TODO @Z:: Log or something
                }
            }
        }
    }

    void update_slots() {
        batch.n_tokens = 0;
        update_prompt_slots();
        update_gen_slots();
    }

    void run() {
        while (!should_exit) {
            process_tasks();
            update_slots();

            bool all_idle = true;
            for (const auto& slot : slots) {
                if (slot.is_processing()) {
                    all_idle = false;
                    break;
                }
            }

            if (all_idle) {
                std::unique_lock lock(mutex_tasks);
                if (queue_tasks.empty()) {
                    cv_tasks.wait(lock, [this]() {
                        return !queue_tasks.empty() || should_exit;
                    });
                }
            }
        }
    }

public:
    Processor(llama_model* model, llama_context* ctx, const int num_slots = 4)
        : model(model), ctx(ctx), tokenizer(model, ctx) {

        batch_size = llama_n_batch(ctx);
        batch = llama_batch_init(static_cast<int32_t>(batch_size), 0, 1);

        slots.reserve(num_slots);
        for (int i = 0; i < num_slots; i++) {
            slots.emplace_back(model, ctx);
            slots.back().job_index = (++current_job_index);
            slots.back().clear(ctx);
        }

        worker_thread = std::thread(&Processor::run, this);
        auto inference_abort_callback = [](void* data) -> bool {
            // Abort inference and reset the abort toggle.
            const auto abort_flag = static_cast<bool*>(data);
            if (*abort_flag) {
                *abort_flag = false;
                return true;
            }
            return false;
        };
        llama_set_abort_callback(ctx, inference_abort_callback, &abort_inference);
    }

    ~Processor() {
        should_exit = true;
        cv_tasks.notify_all();
        if (worker_thread.joinable()) {
            worker_thread.join();
        }
        llama_batch_free(batch);
    }

    //TODO:: @Z: Should this output the cancelled residual outputs or not?
    bool cancel_work(const int request_id_to_cancel) {
        bool found = false;

        // Is our job pending in the request queue? If so, remove it.
        // TODO:: @Z Does a different data structure make more sense with this operation?
        {
            std::lock_guard lock(mutex_tasks);
            if (!queue_tasks.empty()) {
                std::queue<Request> new_queue;

                while (!queue_tasks.empty()) {
                    Request req = queue_tasks.front();
                    queue_tasks.pop();

                    if (req.id != request_id_to_cancel) {
                        new_queue.push(req);
                    } else {
                        if (req.readback_buffer) {
                            readback_finish(req.readback_buffer, R"({"status": "Job Cancelled", "reason": "User requested cancellation"})");
                        }
                        found = true;
                    }
                }

                queue_tasks = std::move(new_queue);
            }
        }

        bool were_any_cancelled = false;

        // Check all slots for the job just in case of a race.
        for (auto& slot : slots) {
            if (slot.request_id == request_id_to_cancel) {
                if (slot.readback_buffer) {
                    readback_finish(slot.readback_buffer, R"({"status": "Job Cancelled", "reason": "User requested cancellation"})");
                }
                slot.clear(ctx);
                slot.job_index = ++current_job_index;
                found = true;
                were_any_cancelled = true;
            }
        }

        //We do not want to throw an abort request cancellation if nothing was cancelled in the processor
        if (!were_any_cancelled) { return false; }

        bool all_idle = true;
        for (auto& slot : slots) {
            if (slot.is_processing()) {
                all_idle = false;
            }
        }

        if (queue_tasks.empty() && all_idle) {
            // Abort inference is reset via the mechanism in the lambda abort fn
            abort_inference = true;
        }
        return found;
    }

    int submit_work(
        const std::string& prompt,
        const InferenceArgs& args,
        ReadbackBuffer* readback_buffer) {
        const auto tokens_opt = tokenizer.tokenize(prompt, args.max_tokens_to_gen);
        if (!tokens_opt) {
            std::cerr << "error: failed to tokenize prompt" << std::endl;
            return -1;
        }
        const std::vector<llama_token>& prompt_tokens = tokens_opt.value();
        static int next_id = 1;
        const int request_id = next_id++;

        {
            const Request request{request_id, prompt_tokens, args, readback_buffer};
            std::lock_guard lock(mutex_tasks);
            queue_tasks.push(request);
        }

        cv_tasks.notify_one();
        return request_id;
    }
};

#endif // PROCESSOR_HPP