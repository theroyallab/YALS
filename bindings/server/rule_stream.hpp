#ifndef RULE_STREAM_HPP
#define RULE_STREAM_HPP

#include "slot.hpp"
#include "sequence_stream.hpp"

#include "llama.h"
#include "sampling.h"
#include <unordered_map>
#include <utility>
#include <vector>
#include <string>
#include <functional>
#include <variant>

enum class TriggerState {
    INACTIVE,
    ACTIVE,
    COMPLETED
};

struct RuleContext {
    const llama_token current_token;
    const SequenceStream::SequenceContext& sequence_ctx;
    SequenceStream* sequence_stream;

    explicit RuleContext(const llama_token token, const SequenceStream::SequenceContext& seq_ctx, SequenceStream* seq_stream)
        : current_token(token), sequence_ctx(seq_ctx), sequence_stream(seq_stream) {
    }
};

struct TriggerOnToken {
    llama_token token;

    explicit TriggerOnToken(const llama_token t) : token(t) {}

    void initialize(const llama_model*, const llama_context*, const Slot&, const RuleContext*) {
    }

    bool should_activate(const llama_model*, const llama_context*, const Slot&, const RuleContext* const rctx) {
        if (!rctx) return false;
        return rctx->current_token == token;
    }
};

struct TriggerOnTokenCount {
    int threshold;

    explicit TriggerOnTokenCount(const int n) : threshold(n) {}

    void initialize(const llama_model*, const llama_context*, const Slot&, const RuleContext*) {
    }

    bool should_activate(const llama_model*, const llama_context*, const Slot& slot, const RuleContext* const) {
        return slot.tokens_generated >= threshold;
    }
};

struct TriggerAlways {
    void initialize(const llama_model*, const llama_context*, const Slot&, const RuleContext*) {
    }

    bool should_activate(const llama_model*, const llama_context*, const Slot&, const RuleContext* const) {
        return true;
    }
};

struct TriggerNever {
    void initialize(const llama_model*, const llama_context*, const Slot&, const RuleContext*) {
    }

    bool should_activate(const llama_model*, const llama_context*, const Slot&, const RuleContext* const) {
        return false;
    }
};

struct TriggerOnSequences {
    std::vector<std::string> sequences;
    std::vector<int> match_ids;
    bool latches_until_reset;
    bool activated = false;
    bool registered = false;

    explicit TriggerOnSequences(
        const std::vector<std::string> &sequences,
        const bool should_latch_until_reset = true):
    sequences(sequences), latches_until_reset(should_latch_until_reset) {
    }

    void register_sequences(SequenceStream* stream) {
        if (!stream || registered) return;

        std::cout << "Registered: ";
        match_ids.clear();
        for (const auto& seq : sequences) {
            int match_id = stream->bind_sequence({seq});
            std::cout << seq << " == ";
            match_ids.push_back(match_id);
        }
        std::cout << std::endl;
        registered = true;
    }

    void unregister_sequences(SequenceStream* stream) {
        if (!stream || !registered) return;

        for (int match_id : match_ids) {
            stream->remove_sequence(match_id);
        }
        match_ids.clear();
        registered = false;
    }

    void initialize(const llama_model*, const llama_context*, const Slot&, const RuleContext* const rctx) {
        if (rctx && rctx->sequence_stream) {
            register_sequences(rctx->sequence_stream);
        }
    }

    bool should_activate(const llama_model*, const llama_context*, const Slot&, const RuleContext* const rctx) {
       if (!rctx || activated) return false;

        for (int match_id : match_ids) {
            if (rctx->sequence_ctx.matched_ids.count(match_id)) {
                activated = true;
                return true;
            }
        }

        return false;
    }

    void reset() {
        activated = false;
    }
};

using Trigger = std::variant<
    TriggerOnToken,
    TriggerOnTokenCount,
    TriggerAlways,
    TriggerNever,
    TriggerOnSequences
>;

struct ActionApplyGrammar {
    std::string grammar;

    explicit ActionApplyGrammar(std::string g) : grammar(std::move(g)) {}

    void start(const llama_model* model, llama_context*, Slot& slot, const RuleContext* const) {
        if (slot.rule_chain) {
            llama_sampler_free(slot.rule_chain);
        }

        slot.rule_chain = llama_sampler_init_llg(
            llama_model_get_vocab(model),
            "lark",
            grammar.c_str()
        );
    }

    void running(const llama_model*, llama_context*, Slot&, const RuleContext* const) {
    }

    void end(const llama_model*, llama_context*, Slot& slot, const RuleContext* const) {
        if (slot.rule_chain) {
            llama_sampler_free(slot.rule_chain);
            slot.rule_chain = nullptr;
        }
    }
};

struct ActionRecordToCallback {
    std::string buffer {};
    std::function<void(std::string)> callback;
    unsigned sequence_status_accept_on_flags;

    explicit ActionRecordToCallback(
        std::function<void(std::string)> callback,
        const unsigned sequence_status_accept_on_flags)
    : callback(std::move(callback)), sequence_status_accept_on_flags(sequence_status_accept_on_flags) {
    }

    void start(const llama_model*, llama_context*, Slot&, const RuleContext* const) {
    }

    void running(const llama_model*, llama_context*, Slot&, const RuleContext* const rctx) {
        if (rctx && rctx->sequence_ctx.sequence_status & sequence_status_accept_on_flags) {
            buffer += rctx->sequence_ctx.current_sequence;
        }
    }

    void end(const llama_model*, llama_context*, Slot&, const RuleContext* const rctx) {
        if (rctx && rctx->sequence_ctx.sequence_status & sequence_status_accept_on_flags) {
            buffer += rctx->sequence_ctx.current_text_piece;
        } else if (rctx)
        {
            buffer += rctx->sequence_ctx.unmatched_sequence;
        }
        callback(std::move(buffer));
        buffer = "";
    }
};

struct ActionBanStopTokens {
    void start(const llama_model* model, llama_context*, Slot& slot, const RuleContext* const) {
        const std::vector terminal_token_bans {llama_vocab_eos(llama_model_get_vocab(model)), llama_vocab_eot(llama_model_get_vocab(model))};
        slot.presampler.add_eos_ban(model, terminal_token_bans);
    }

    void running(const llama_model*, llama_context*, Slot&, const RuleContext* const) {
    }

    void end(const llama_model* model, llama_context*, Slot& slot, const RuleContext* const&) {
        slot.presampler.clear_eos_bans(model);
    }
};

struct ActionEndGeneration {
    std::string stop_reason;

    explicit ActionEndGeneration(std::string reason) : stop_reason(std::move(reason)) {}

    void start(const llama_model*, llama_context*, Slot&, const RuleContext* const) {
    }

    void running(const llama_model*, llama_context*, Slot&, const RuleContext* const) {
    }

    void end(const llama_model*, llama_context*, Slot&, const RuleContext* const) {
    }
};

using Action = std::variant<
    ActionApplyGrammar,
    ActionBanStopTokens,
    ActionEndGeneration,
    ActionRecordToCallback
>;

struct Rule {
    Trigger start_trigger;
    Trigger end_trigger;
    std::vector<Action> actions;
    TriggerState state = TriggerState::INACTIVE;
    bool reusable;

    Rule(Trigger start, Trigger end, std::vector<Action> a, const bool can_reuse = false)
        : start_trigger(std::move(start)), end_trigger(std::move(end)), actions(std::move(a)), reusable(can_reuse) {}

    Rule(Trigger start, Trigger end, Action a, const bool can_reuse = false)
        : start_trigger(std::move(start)), end_trigger(std::move(end)), actions{std::move(a)}, reusable(can_reuse) {}

    void cleanup_sequences(SequenceStream* stream) {
        auto cleanup_trigger = [stream](Trigger& trigger) {
            if (auto* seq_trigger = std::get_if<TriggerOnSequences>(&trigger)) {
                seq_trigger->unregister_sequences(stream);
            }
        };

        cleanup_trigger(start_trigger);
        cleanup_trigger(end_trigger);
    }

    std::vector<std::reference_wrapper<const Action>> process(const llama_model* model, llama_context* ctx, Slot& slot, const RuleContext* const context) {
        const TriggerState prev_state = state;
        std::vector<std::reference_wrapper<const Action>> completed_actions;

        if (state == TriggerState::INACTIVE) {
            std::visit([&](auto& t) { t.initialize(model, ctx, slot, context); }, start_trigger);

            const bool should_activate = std::visit([&](auto& t) -> bool {
                return t.should_activate(model, ctx, slot, context);
            }, start_trigger);

            if (should_activate) {
                state = TriggerState::ACTIVE;
                std::visit([&](auto& t) { t.initialize(model, ctx, slot, context); }, end_trigger);
            }
        }
        else if (state == TriggerState::ACTIVE) {
            const bool should_complete = std::visit([&](auto& t) -> bool {
                return t.should_activate(model, ctx, slot, context);
            }, end_trigger);

            if (should_complete) {
                state = TriggerState::COMPLETED;
            }
        }

        if (prev_state == TriggerState::INACTIVE && state == TriggerState::ACTIVE) {
            for (auto& action : actions) {
                std::visit([&](auto& a) { a.start(model, ctx, slot, context); }, action);
            }
        }
        else if (state == TriggerState::ACTIVE) {
            for (auto& action : actions) {
                std::visit([&](auto& a) { a.running(model, ctx, slot, context); }, action);
            }
        }
        else if (prev_state == TriggerState::ACTIVE && state == TriggerState::COMPLETED) {
            for (auto& action : actions) {
                std::visit([&](auto& a) { a.end(model, ctx, slot, context); }, action);
            }
        }

        if (prev_state == TriggerState::ACTIVE && state == TriggerState::COMPLETED) {
            for (const auto& action : actions) {
                completed_actions.push_back(action);
            }

            if (reusable) {
                state = TriggerState::INACTIVE;
                auto reset_trigger = [](Trigger& trigger) {
                    if (auto* seq_trigger = std::get_if<TriggerOnSequences>(&trigger)) {
                        seq_trigger->reset();
                    }
                };
                reset_trigger(start_trigger);
                reset_trigger(end_trigger);
            }
        }

        return completed_actions;
    }
};

class RuleStream {
    std::unordered_map<unsigned, std::vector<Rule>> rules_by_id;
    unsigned current_id = 0;

    std::vector<std::reference_wrapper<const Action>> process_rules(
        const RuleContext* const context,
        const llama_model* model,
        llama_context* ctx,
        Slot& slot,
        std::vector<Rule>& rules
    ) {
        std::vector<std::reference_wrapper<const Action>> triggered_actions;

        for (auto& rule : rules) {
            auto actions = rule.process(model, ctx, slot, context);
            triggered_actions.insert(triggered_actions.end(), actions.begin(), actions.end());
        }

        return triggered_actions;
    }

public:
    void initialize_all_triggers(SequenceStream* sequence_stream, const llama_model* model, llama_context* ctx, Slot& slot) {
        SequenceStream::SequenceContext dummy_seq_ctx{};
        RuleContext dummy_context{0, dummy_seq_ctx, sequence_stream};

        for (auto& [id, rule_list] : rules_by_id) {
            for (auto& rule : rule_list) {
                std::visit([&](auto& t) { t.initialize(model, ctx, slot, &dummy_context); }, rule.start_trigger);
                std::visit([&](auto& t) { t.initialize(model, ctx, slot, &dummy_context); }, rule.end_trigger);
            }
        }
    }

    unsigned add_rules(std::vector<Rule> rules,
                      const llama_model* model,
                      llama_context* ctx,
                      Slot& slot) {

        const unsigned rule_id = current_id++;
        rules_by_id[rule_id] = std::move(rules);

        process_rules(nullptr, model, ctx, slot, rules_by_id[rule_id]);
        return rule_id;
    }

    void remove_id(const unsigned id, SequenceStream* stream = nullptr) {
        auto it = rules_by_id.find(id);
        if (it != rules_by_id.end()) {
            if (stream) {
                for (auto& rule : it->second) {
                    rule.cleanup_sequences(stream);
                }
            }
            rules_by_id.erase(it);
        }
    }

    const std::vector<Rule>* get_rules(const unsigned id) const {
        if (const auto it = rules_by_id.find(id); it != rules_by_id.end()) {
            return &it->second;
        }
        return nullptr;
    }

    std::vector<std::reference_wrapper<const Action>> apply_engine(
        const llama_token token,
        const SequenceStream::SequenceContext& seq_ctx,
        SequenceStream* sequence_stream,
        const llama_model* model,
        llama_context* ctx,
        Slot& slot
    ) {
        std::vector<std::reference_wrapper<const Action>> all_triggered_actions;
        const RuleContext context_obj{token, seq_ctx, sequence_stream};
        const RuleContext* const context = &context_obj;

        for (auto& [id, rule_list] : rules_by_id) {
            auto triggered_actions = process_rules(context, model, ctx, slot, rule_list);
            all_triggered_actions.insert(
                all_triggered_actions.end(),
                triggered_actions.begin(),
                triggered_actions.end()
            );
        }

        return all_triggered_actions;
    }

    void reset(SequenceStream* stream = nullptr) {
        if (stream) {
            for (auto& [id, rule_list] : rules_by_id) {
                for (auto& rule : rule_list) {
                    rule.cleanup_sequences(stream);
                }
            }
        }
        rules_by_id.clear();
        current_id = 0;
    }
};

namespace RuleEngine {

inline unsigned rule_max_tokens(
    RuleStream& stream, const int num_tokens,
    const llama_model* model, llama_context* ctx, Slot& slot
) {
    std::vector<Rule> rules;
    Action end_action = ActionEndGeneration("MaxNewTokens");
    rules.emplace_back(TriggerOnTokenCount(num_tokens), TriggerAlways(), end_action);
    return stream.add_rules(std::move(rules), model, ctx, slot);
}

inline unsigned rule_stop_tokens(
    RuleStream& stream, const std::vector<int32_t>& stopping_tokens,
    const llama_model* model, llama_context* ctx, Slot& slot
) {
    std::vector<Rule> stopping_rules;
    for (const auto& stopping_token : stopping_tokens) {
        Action end_action = ActionEndGeneration("StopToken");
        stopping_rules.emplace_back(TriggerOnToken(stopping_token), TriggerAlways(), end_action);
    }
    return stream.add_rules(std::move(stopping_rules), model, ctx, slot);
}

inline unsigned rule_min_tokens(
    RuleStream& stream, const int num_tokens,
    const llama_model* model, llama_context* ctx, Slot& slot
) {
    std::vector<Rule> rules;
    Action ban_action = ActionBanStopTokens();
    rules.emplace_back(TriggerAlways(), TriggerOnTokenCount(num_tokens), ban_action);
    return stream.add_rules(std::move(rules), model, ctx, slot);
}

inline unsigned rule_constrain_grammar(
    RuleStream& stream, const std::string& grammar,
    const llama_token apply_token, const llama_token remove_token,
    const llama_model* model, llama_context* ctx, Slot& slot
) {
    std::vector<Rule> rules;
    Action grammar_action = ActionApplyGrammar(grammar);
    rules.emplace_back(TriggerOnToken(apply_token), TriggerOnToken(remove_token), grammar_action);
    return stream.add_rules(std::move(rules), model, ctx, slot);
}

inline unsigned rule_complex_action(
    RuleStream& stream, const Trigger& start, const Trigger& end,
    const std::vector<Action>& actions,
    const llama_model* model, llama_context* ctx, Slot& slot,
    bool reusable
) {
    std::vector<Rule> rules;
    rules.emplace_back(start, end, actions, reusable);
    return stream.add_rules(std::move(rules), model, ctx, slot);
}

inline unsigned rule_record_constrained_grammar(
    RuleStream& stream,
    const std::string& grammar,
    const std::function<void(std::string)> &callback,
    const llama_model* model, llama_context* ctx, Slot& slot
    ) {
    std::vector<Action> actions {
        ActionApplyGrammar(grammar),
        ActionRecordToCallback(callback, SequenceStream::ACCEPT)
    };
    return rule_complex_action(stream, TriggerOnSequences({"Hello"}), TriggerOnSequences({"}"}), std::move(actions), model, ctx, slot, true);
}

}

#endif