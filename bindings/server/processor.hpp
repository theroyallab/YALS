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
#include <thread>

#include "inference_args.hpp"
#include "llama.h"
#include "tokenization.hpp"
#include "slot.hpp"
#include "request.hpp"
#include "sequence_stream.hpp"
#include "json_status.hpp"
#include "rule_stream.hpp"

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

template<class... Ts> struct rule_action_type : Ts... { using Ts::operator()...; };
template<class... Ts> rule_action_type(Ts...) -> rule_action_type<Ts...>;

class Processor {
    llama_model* model;
    llama_context* ctx;
    llama_memory_t mem;
    llama_batch batch{};
    bool abort_inference = false;

    std::vector<Slot> slots;
    uint32_t batch_size;

    std::queue<Request> queue_tasks;
    std::mutex mutex_tasks;
    std::condition_variable cv_tasks;

    std::thread worker_thread;
    std::atomic<bool> should_exit{false};

    std::atomic<int> current_job_index = 0;
    Tokenizer tokenizer;

    // nearly eq to common_add_to_batch from lcpp server
    void add_to_batch(Slot& slot, const llama_token token, const bool compute_logits) {
        slot.i_batch = batch.n_tokens;

        batch.token[batch.n_tokens] = token;
        batch.pos[batch.n_tokens] = slot.n_past;
        batch.n_seq_id[batch.n_tokens] = 1;
        batch.seq_id[batch.n_tokens][0] = slot.slot_id;
        batch.logits[batch.n_tokens] = static_cast<int8_t>(compute_logits);

        batch.n_tokens++;
        slot.n_past++;
    }

    static double readable_ggml_time() {
        return static_cast<double>(ggml_time_us()) * 1e-3;
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

        // Cleanup cancelled slots
        // TODO: This is not optimal due to the extra for loop
        for (auto& slot : slots) {
            if (slot.cancelled) {
                cleanup_slot(slot);
            }
        }

        // Check if an idle slot is present
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
            inference_args] = queue_tasks.front();

        queue_tasks.pop();
        lock.unlock();

        // Prompt + max tokens to gen is longer than the entire ctx length.
        const auto total_tokens = prompt_tokens.size() + inference_args.max_tokens_to_gen;
        if (total_tokens > llama_n_ctx(ctx) || total_tokens > inference_args.max_slot_n_ctx) {
            readback_finish(inference_args.gen_resources->readback_buffer, make_empty_json_status_string("CtxExceeded", "None"));
            return;
        }

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
            llama_memory_seq_rm(mem, best_slot->slot_id, longest_prefix, -1);

            best_slot->prompt_tokens_processed = longest_prefix;
            best_slot->n_past = longest_prefix;
            best_slot->last_token = prompt_tokens[longest_prefix - 1];

            best_slot->state =
                longest_prefix == prompt_tokens.size() ?
                Slot::State::GENERATING :
                Slot::State::PROMPT;
        } else {
            llama_memory_seq_rm(mem, best_slot->slot_id, 0, -1);
            best_slot->prompt_tokens_processed = 0;
            best_slot->state = Slot::State::PROMPT;
            best_slot->prompt_tokens.clear();
        }

        best_slot->request_id = id;
        best_slot->prompt_tokens = prompt_tokens;

        if (best_slot->gen_resources) {
            generation_resources_release(best_slot->gen_resources);
        }
        best_slot->gen_resources = generation_resources_ref_acquire(inference_args.gen_resources);

        best_slot->slot_start_time = readable_ggml_time();

        best_slot->sequence_stream->bind_sequences(inference_args.stopping_strings, inference_args.rewind_strings);
        best_slot->rewind_snapshot = Slot::SlotSnapshot::snapshot_slot(*best_slot, mem, false);

        best_slot->sampler = best_slot->gen_resources->sampler;
        best_slot->n_ctx_max = inference_args.max_slot_n_ctx;

        if (inference_args.min_tokens_to_gen > 0) {
            RuleEngine::rule_min_tokens(*best_slot->rule_stream, inference_args.min_tokens_to_gen, model, ctx, *best_slot);
        }
        
        if (inference_args.max_tokens_to_gen > 0 && inference_args.max_tokens_to_gen >= inference_args.min_tokens_to_gen) {
            RuleEngine::rule_max_tokens(*best_slot->rule_stream, inference_args.max_tokens_to_gen, model, ctx, *best_slot);
        }
    }

    // Processes the next sequence token. Finalizes the request if gen is finished.
    bool process_token(Slot& slot, const llama_token token) const {

        // Decode special sets parse_special for decoding ONLY
        auto piece = slot.detokenizer->process_token(token, true);
        const bool is_eos = tokenizer.is_end_of_generation_token(token);
        bool is_complete = is_eos;
        bool yield_final = false;

        slot.tokens_generated++;

        std::string finish_reason = "Unspecified";
        std::string stop_token = "Unspecified";

        if (is_eos) {
            finish_reason = "StopToken";
            stop_token = common_token_to_piece(ctx, token, true);
        }

        if (llama_memory_seq_pos_max(mem, slot.slot_id) >= slot.n_ctx_max || llama_memory_seq_pos_max(mem, slot.slot_id) >= llama_n_ctx(ctx)) {
            is_complete = true;
            finish_reason = "CtxExceeded";
            stop_token = common_token_to_piece(ctx, token, true);
        }

        const auto seq_res = slot.sequence_stream->append(piece);
        const auto triggered_actions = slot.rule_stream->apply_engine(token, seq_res, model, ctx, slot);
        for (const auto& actionWrapper : triggered_actions) {
            std::visit(rule_action_type {

                //case: ActionEndGeneration:
                [&](const ActionEndGeneration& action) {
                    finish_reason = action.stop_reason;
                    is_complete = true;
                },

                //default: (does nothing)
                [](auto&&) { }

            }, actionWrapper.get());
        }

        switch (seq_res.sequence_status) {
            case SequenceStream::SequenceStatus::ACCEPT:
                if (!seq_res.current_sequence.empty() && !is_eos) {
                    slot.generated_text += seq_res.current_sequence;
                    readback_write_to_buffer(slot.gen_resources->readback_buffer, seq_res.current_sequence, token);
                }

                slot.presampler.clear_rewind_bans(model);
                slot.rewind_snapshot = Slot::SlotSnapshot::snapshot_slot(slot, mem, false);
                break;
            case SequenceStream::SequenceStatus::REWIND: {
                //Restore the slot to whatever the last accepted snapshot was.
                //Then delete the part of the KV we're rewinding
                const int32_t prev_kv_pos = slot.rewind_snapshot.rewind_slot(slot);
                llama_memory_seq_rm(mem, slot.slot_id, prev_kv_pos, -1);

                //Ban every token in the buffer.
                const auto tokens = tokenizer.tokenize(seq_res.current_sequence, false, false);
                slot.presampler.add_rewind_bans(model, tokens);

                return true;
            }
            case SequenceStream::SequenceStatus::STOP:
                is_complete = true;
                finish_reason = "StopString";
                stop_token = seq_res.current_sequence;
                piece = seq_res.unmatched_sequence;

                // Write the unmatched sequence to buffer
                if (!seq_res.unmatched_sequence.empty()) {
                    slot.generated_text += seq_res.unmatched_sequence;
                    readback_write_to_buffer(slot.gen_resources->readback_buffer, seq_res.unmatched_sequence, token);
                }

                break;
            case SequenceStream::SequenceStatus::BUFFER:
                break;
        }

        if (!is_complete) {
            return !is_eos;
        }

        // Write any remaining text from detokenizer
        if (slot.detokenizer->has_incomplete()) {
            const std::string remaining = slot.detokenizer->flush();

            if (!remaining.empty() && !is_eos) {
                slot.generated_text += remaining;
                readback_write_to_buffer(slot.gen_resources->readback_buffer, remaining, token);
            }
        }

        slot.generating_end_time = readable_ggml_time();
        const auto status = make_json_status_string(slot, finish_reason, stop_token);
        readback_finish(slot.gen_resources->readback_buffer, status);
        return false;
    }

    void update_batch() {
        batch.n_tokens = 0;
        for (auto& slot : slots) {
            if (slot.is_processing_prompt() && slot.prompt_tokens_processed < slot.prompt_tokens.size()) {
                while (batch.n_tokens < batch_size) {

                    const llama_token token = slot.prompt_tokens[slot.prompt_tokens_processed];
                    const bool is_last_prompt_token = (slot.prompt_tokens_processed == slot.prompt_tokens.size() - 1);
                    slot.i_batch = batch.n_tokens;
                    slot.prompt_tokens_processed++;
                    slot.last_token = token;
                    add_to_batch(slot, token, is_last_prompt_token);

                    if (slot.prompt_tokens_processed >= slot.prompt_tokens.size()) {
                        slot.state = Slot::State::GENERATING;
                        slot.rewind_snapshot = Slot::SlotSnapshot::snapshot_slot(slot, mem, true);
                        break;
                    }
                }
            } else {
                if (slot.is_generating() && batch.n_tokens < batch_size) {
                    add_to_batch(slot, slot.last_token, true);
                }
            }
        }
    }

    [[nodiscard]] llama_token sample(const Slot& slot) const {
        if (slot.presampler.sampler) {
            const auto pre_n = llama_sampler_chain_n(slot.presampler.sampler);
            llama_sampler_chain_add(slot.presampler.sampler, slot.sampler);
            const auto token = llama_sampler_sample(slot.presampler.sampler, ctx, slot.i_batch);

            while (llama_sampler_chain_n(slot.presampler.sampler) > pre_n) {
                llama_sampler_chain_remove(slot.presampler.sampler, pre_n);
            }
            return token;
        }

        return llama_sampler_sample(slot.sampler, ctx, slot.i_batch);
    }

    void update_gen_slots() {
        if (batch.n_tokens == 0) {
            return;
        }

        while (true) {
            const int32_t decode_result = llama_decode(ctx, batch);

            //Decode aborted, this is not a failure, we can redo the decode.
            if (decode_result == 2) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }

            //TODO:: @Z We can potentially avoid a hard abort depending on the status code. Investigate if possibel.
            if (decode_result != 0) {
                for (auto& slot : slots) {
                    if (slot.i_batch >= 0 && slot.i_batch < batch.n_tokens) {
                        slot.generating_end_time = readable_ggml_time();
                        readback_finish(slot.gen_resources->readback_buffer, make_json_status_string(slot, "BatchDecode", ""));
                        cleanup_slot(slot);
                    }
                }
                return;
            }
            break;
        }

        for (auto& slot : slots) {
            // Do nothing if slot isn't part of the current batch
            if (slot.i_batch < 0 || slot.i_batch >= batch.n_tokens) {
                continue;
            }

            if (slot.is_generating()) {
                // Triggered right when generation starts = prompt process ended
                if (slot.prompt_end_time == 0.0) {
                    slot.prompt_end_time = readable_ggml_time();
                }

                const llama_token token = sample(slot);
                slot.last_token = token;
                slot.i_batch = -1;

                if (const bool continue_gen = process_token(slot, token); !continue_gen) {
                    cleanup_slot(slot);
                    //Status reported by process_token
                }
            }
        }
    }

    void update_slots() {
        common_batch_clear(batch);

        update_batch();
        update_gen_slots();
    }

    // Required due to rule_stream circular dependency
    void cleanup_slot(Slot& slot) {
        slot.rule_stream->reset();
        slot.end(++current_job_index, ctx);
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
    Processor(llama_model* model, llama_context* ctx, llama_memory_t mem, const int num_slots = 4)
        : model(model), ctx(ctx), mem(mem), tokenizer(model, ctx) {

        batch_size = llama_n_batch(ctx);
        batch = llama_batch_init(static_cast<int32_t>(batch_size), 0, num_slots);

        slots.reserve(num_slots);
        for (int i = 0; i < num_slots; i++) {
            slots.emplace_back(model, ctx);
            slots.back().end(++current_job_index, ctx);
            slots.back().slot_id = i;
            slots.back().rule_stream = new RuleStream();
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
        for (const auto& slot : slots) {
            delete slot.rule_stream;
        }
        if (worker_thread.joinable()) {
            worker_thread.join();
        }
        llama_batch_free(batch);
    }

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
                        readback_finish(
                            req.inference_args.gen_resources->readback_buffer,
                            make_empty_json_status_string("Aborted", "None")
                        );
                        found = true;
                    }
                }

                queue_tasks = std::move(new_queue);
            }
        }

        bool were_any_cancelled = false;

        for (auto& slot : slots) {
            if (slot.request_id == request_id_to_cancel) {
                if (slot.gen_resources->readback_buffer) {
                    std::string last_token_piece = common_token_to_piece(ctx, slot.last_token, true);
                    slot.generating_end_time = readable_ggml_time();
                    readback_finish(slot.gen_resources->readback_buffer, make_json_status_string(slot, "Aborted", last_token_piece));
                }
                
                // Mark for cancellation rather than direct cleanup
                slot.cancelled = true;

                found = true;
                were_any_cancelled = true;
            }
        }

        //We do not want to throw an abort request cancellation if nothing was cancelled in the processor
        if (!were_any_cancelled) {
          return false;
        }

        // A cancelled slot is "idle"
        bool all_idle = true;
        for (auto& slot : slots) {
            if (slot.is_processing() && !slot.cancelled) {
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
        const InferenceArgs& args) {

        // Always encode special tokens
        const std::vector<llama_token>& prompt_tokens = tokenizer.tokenize(prompt, args.add_special, true);
        static int next_id = 1;
        const int request_id = next_id++;

        {
            const Request request{request_id, prompt_tokens, args};
            std::lock_guard lock(mutex_tasks);
            queue_tasks.push(request);
        }

        cv_tasks.notify_one();
        return request_id;
    }
};

#endif // PROCESSOR_HPP