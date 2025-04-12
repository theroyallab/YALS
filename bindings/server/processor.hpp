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

        // Prompt is longer than the entire ctx length.
        if (prompt_tokens.size() > llama_n_ctx(ctx) || prompt_tokens.size() > inference_args.max_slot_n_ctx) {
            readback_finish(readback_buffer, make_empty_json_status_string("CtxExceeded", "None"));
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
            llama_kv_self_seq_rm(ctx, best_slot->slot_id, longest_prefix, -1);

            best_slot->prompt_tokens_processed = longest_prefix;
            best_slot->n_past = longest_prefix;
            best_slot->last_token = prompt_tokens[longest_prefix - 1];

            best_slot->state =
                longest_prefix == prompt_tokens.size() ?
                Slot::State::GENERATING :
                Slot::State::PROMPT;
        } else {
            llama_kv_self_seq_rm(ctx, best_slot->slot_id, 0, -1);
            best_slot->prompt_tokens_processed = 0;
            best_slot->state = Slot::State::PROMPT;
            best_slot->prompt_tokens.clear();
        }

        best_slot->request_id = id;
        best_slot->prompt_tokens = prompt_tokens;

        best_slot->resource_bundle = resource_bundle_ref_acquire(inference_args.resource_bundle);

        best_slot->slot_start_time = 1e-3 * ggml_time_us();

        best_slot->sequence_stream->bind_sequences(inference_args.stopping_strings, inference_args.rewind_strings);
        best_slot->rewind_snapshot = Slot::SlotSnapshot::snapshot_slot(*best_slot, ctx, false);

        best_slot->multi_sampler.sampler = best_slot->resource_bundle->sampler;
        best_slot->n_ctx_max = inference_args.max_slot_n_ctx;

        // for (const auto& apply_rule : rules) {
        //     apply_rule(*best_slot->rule_stream, model, ctx, *best_slot);
        // }

        if (inference_args.min_tokens_to_gen > 0) {
            RuleEngine::rule_min_tokens(*best_slot->rule_stream, inference_args.min_tokens_to_gen, model, ctx, *best_slot);
        }
        
        if (inference_args.max_tokens_to_gen > 0 && inference_args.max_tokens_to_gen >= inference_args.min_tokens_to_gen) {
            RuleEngine::rule_max_tokens(*best_slot->rule_stream, inference_args.max_tokens_to_gen, model, ctx, *best_slot);
        }
    }

    void defrag_kv_if_thresh_greater(const float thresh) const {
        const auto used_cells = llama_kv_self_used_cells(ctx);
        const auto total_cells = llama_n_ctx(ctx);
        if (static_cast<float>(used_cells) > thresh * static_cast<float>(total_cells) && slots.size() > 1) {
            llama_kv_self_defrag(ctx);
        }
    }

    // Processes the next sequence token. Finalizes the request if gen is finished.
    bool process_token(Slot& slot, const llama_token token) const {
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

        if (llama_kv_self_seq_pos_max(ctx, slot.slot_id) >= slot.n_ctx_max || llama_kv_self_seq_pos_max(ctx, slot.slot_id) >= llama_n_ctx(ctx)) {
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
            case SequenceStream::SequenceStatus::ACCEPT: {
                slot.generated_text += seq_res.current_sequence;
                slot.multi_sampler.presampler.clear_rewind_bans(model);
                slot.rewind_snapshot = Slot::SlotSnapshot::snapshot_slot(slot, ctx, false);
                yield_final = true;
                }
                break;
            case SequenceStream::SequenceStatus::REWIND: {
                //Restore the slot to whatever the last accepted snapshot was.
                //Then delete the part of the KV we're rewinding
                const int32_t prev_kv_pos = slot.rewind_snapshot.rewind_slot(slot);
                llama_kv_self_seq_rm(ctx, slot.slot_id, prev_kv_pos, -1);

                //Ban every token in the buffer.
                const auto tokens = tokenizer.tokenize(seq_res.current_sequence, false, false);
                slot.multi_sampler.presampler.add_rewind_bans(model, tokens);

                }
                return true;
            case SequenceStream::SequenceStatus::STOP:
                is_complete = true;
                finish_reason = "StopString";
                stop_token = seq_res.current_sequence;
                piece = seq_res.unmatched_sequence;
                yield_final = true;
                break;
            case SequenceStream::SequenceStatus::BUFFER:
                break;
        }


        if (!is_complete) {
            if (!piece.empty()) {
                readback_write_to_buffer(slot.resource_bundle->readback_buffer, piece, token);
            }
            return !is_eos;
        }

        std::string final_piece = piece;
        if (slot.detokenizer->has_incomplete()) {
            const std::string remaining = slot.detokenizer->flush();
            if (!remaining.empty()) final_piece += remaining;
        }

        if (yield_final && !final_piece.empty()) {
            readback_write_to_buffer(slot.resource_bundle->readback_buffer, final_piece, token);
        }

        slot.generating_end_time = 1e-3 * ggml_time_us();
        const auto status = make_json_status_string(slot, finish_reason, stop_token);
        readback_finish(slot.resource_bundle->readback_buffer, status);
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
                        slot.rewind_snapshot = Slot::SlotSnapshot::snapshot_slot(slot, ctx, true);
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
                        slot.generating_end_time = 1e-3 * ggml_time_us();
                        readback_finish(slot.resource_bundle->readback_buffer, make_json_status_string(slot, "BatchDecode", ""));
                        slot.end(++current_job_index, ctx);
                    }
                }
                return;
            }
            break;
        }

        for (auto& slot : slots) {
            if (slot.prompt_end_time == 0.0) {
                slot.prompt_end_time =  1e-3 * ggml_time_us();
            }

            if (slot.i_batch < 0 || slot.i_batch >= batch.n_tokens) {
                continue;
            }

            if (slot.is_generating()) {
                const auto maybe_token = slot.multi_sampler.sample(ctx, slot.i_batch);
                if (!maybe_token.has_value()) {
                    // TODO:: @Z Bug.
                    break;
                }
                const llama_token token = maybe_token.value();
                slot.last_token = token;
                slot.i_batch = -1;

                if (const bool continue_gen = process_token(slot, token); !continue_gen) {
                    slot.end(++current_job_index, ctx);
                    //Status reported by process_token
                }
            }
        }
    }

    void update_slots() {
        common_batch_clear(batch);
        defrag_kv_if_thresh_greater(.9);

        update_batch();
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
                defrag_kv_if_thresh_greater(0.6);
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
                        if (req.readback_buffer) {
                            readback_finish(req.readback_buffer, make_empty_json_status_string("Aborted", "None"));
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
                if (slot.resource_bundle->readback_buffer) {
                    std::string last_token_piece = common_token_to_piece(ctx, slot.last_token, true);
                    slot.generating_end_time = 1e-3 * ggml_time_us();
                    readback_finish(slot.resource_bundle->readback_buffer, make_json_status_string(slot, "Aborted", last_token_piece));
                }
                slot.end(++current_job_index, ctx);
                found = true;
                were_any_cancelled = true;
            }
        }

        //We do not want to throw an abort request cancellation if nothing was cancelled in the processor
        if (!were_any_cancelled) {
          return false;
        }

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
        const InferenceArgs& args) {
        const std::vector<llama_token>& prompt_tokens = tokenizer.tokenize(prompt, args.max_tokens_to_gen);
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