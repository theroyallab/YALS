#ifndef RULE_STREAM_HPP
#define RULE_STREAM_HPP

#include <multisampler.hpp>
#include <slot.hpp>
#include <sequence_stream.hpp>

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

    explicit RuleContext(const llama_token token, const SequenceStream::SequenceContext& seq_ctx)
        : current_token(token), sequence_ctx(seq_ctx) {}
};

struct TriggerOnToken {
    llama_token token;

    explicit TriggerOnToken(const llama_token t) : token(t) {}

    bool should_activate(const llama_model*, const llama_context*, const Slot&, const RuleContext* const rctx) const {
        if (!rctx) return false;
        return rctx->current_token == token;
    }
};

struct TriggerOnTokenCount {
    int threshold;

    explicit TriggerOnTokenCount(const int n) : threshold(n) {}

    bool should_activate(const llama_model*, const llama_context*, const Slot& slot, const RuleContext* const) const {
        return slot.tokens_generated >= threshold;
    }
};

struct TriggerAlways {
    bool should_activate(const llama_model*, const llama_context*, const Slot&, const RuleContext* const) const {
        return true;
    }
};

struct TriggerNever {
    bool should_activate(const llama_model*, const llama_context*, const Slot&, const RuleContext* const) const {
        return false;
    }
};

using Trigger = std::variant<
    TriggerOnToken,
    TriggerOnTokenCount,
    TriggerAlways,
    TriggerNever
>;

struct ActionApplyGrammar {
    std::string grammar;

    explicit ActionApplyGrammar(std::string g) : grammar(std::move(g)) {}

    void start(const llama_model* model, llama_context*, Slot& slot, const RuleContext* const) {
        if (slot.multi_sampler.constraint_sampler) {
            llama_sampler_free(slot.multi_sampler.constraint_sampler);
            slot.multi_sampler.constraint_sampler = nullptr;
        }

        slot.multi_sampler.constraint_sampler = llama_sampler_init_llg(
            llama_model_get_vocab(model),
            "lark",
            grammar.c_str()
        );
    }

    void running(const llama_model*, llama_context*, Slot&, const RuleContext* const) {
    }

    void end(const llama_model*, llama_context*, Slot& slot, const RuleContext* const) {
        if (slot.multi_sampler.constraint_sampler) {
            llama_sampler_free(slot.multi_sampler.constraint_sampler);
            slot.multi_sampler.constraint_sampler = nullptr;
        }
    }
};

struct ActionRecordToCallback {
    std::string buffer {};
    std::function<void(std::string)> callback;
    SequenceStream::SequenceStatus accept_on_flags;

    explicit ActionRecordToCallback(
        std::function<void(std::string)> callback,
        const SequenceStream::SequenceStatus flags)
    : callback(std::move(callback)), accept_on_flags(flags) {
    }

    void start(const llama_model*, llama_context*, Slot&, const RuleContext* const) {
    }

    void running(const llama_model*, llama_context*, Slot&, const RuleContext* const rctx) {
        if (rctx && rctx->sequence_ctx.sequence_status & accept_on_flags) {
            buffer += rctx->sequence_ctx.current_text_piece;
        }
    }

    void end(const llama_model*, llama_context*, Slot&, const RuleContext* const rctx) {
        if (rctx && rctx->sequence_ctx.sequence_status & accept_on_flags) {
            buffer += rctx->sequence_ctx.current_text_piece;
        }
        callback(buffer);
    }
};

struct ActionBanStopTokens {
    void start(const llama_model* model, llama_context*, Slot& slot, const RuleContext* const) {
        const std::vector terminal_token_bans {llama_vocab_eos(llama_model_get_vocab(model)), llama_vocab_eot(llama_model_get_vocab(model))};
        slot.multi_sampler.presampler.add_eos_ban(model, terminal_token_bans);
    }

    void running(const llama_model*, llama_context*, Slot&, const RuleContext* const) {
    }

    void end(const llama_model* model, llama_context*, Slot& slot, const RuleContext* const&) {
        slot.multi_sampler.presampler.clear_eos_bans(model);
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

    Rule(Trigger start, Trigger end, std::vector<Action> a)
        : start_trigger(std::move(start)), end_trigger(std::move(end)), actions(std::move(a)) {}

    Rule(Trigger start, Trigger end, Action a)
        : start_trigger(std::move(start)), end_trigger(std::move(end)), actions{std::move(a)} {}

    std::vector<std::reference_wrapper<const Action>> process(const llama_model* model, llama_context* ctx, Slot& slot, const RuleContext* const context) {
        const TriggerState prev_state = state;
        std::vector<std::reference_wrapper<const Action>> completed_actions;

        if (state == TriggerState::INACTIVE) {
            const bool should_activate = std::visit([&](const auto& t) -> bool {
                return t.should_activate(model, ctx, slot, context);
            }, start_trigger);

            if (should_activate) {
                state = TriggerState::ACTIVE;
            }
        }
        else if (state == TriggerState::ACTIVE) {
            const bool should_complete = std::visit([&](const auto& t) -> bool {
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
        }

        return completed_actions;
    }
};

class RuleStream {
    std::unordered_map<unsigned, std::vector<Rule>> rules_by_id;
    unsigned current_id = 0;

private:
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
    unsigned add_rules(std::vector<Rule> rules,
                      const llama_model* model,
                      llama_context* ctx,
                      Slot& slot) {

        const unsigned rule_id = current_id++;
        rules_by_id[rule_id] = std::move(rules);

        process_rules(nullptr, model, ctx, slot, rules_by_id[rule_id]);
        return rule_id;
    }

    void remove_id(const unsigned id) {
        rules_by_id.erase(id);
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
        const llama_model* model,
        llama_context* ctx,
        Slot& slot
    ) {
        const RuleContext context_obj{token, seq_ctx};
        const RuleContext* const context = &context_obj;
        std::vector<std::reference_wrapper<const Action>> all_triggered_actions;

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

    void reset() {
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
    const llama_model* model, llama_context* ctx, Slot& slot
) {
    std::vector<Rule> rules;
    rules.emplace_back(start, end, actions);
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
    return rule_complex_action(stream, TriggerOnTokenCount(50), TriggerOnToken(1125), std::move(actions), model, ctx, slot);
}

}

#endif