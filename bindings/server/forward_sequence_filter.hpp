#ifndef FORWARD_SEQUENCE_FILTER_HPP
#define FORWARD_SEQUENCE_FILTER_HPP
#include <unordered_map>
#include <string>
#include <vector>
#include <queue>
#include <set>
#include <algorithm>

class ForwardSequenceMatcher {
public:
    struct State {
        std::unordered_map<unsigned char, size_t> goto_transitions;
        std::set<std::string> output;
    };

    bool case_sensitive;
    std::set<std::string> patterns;
    std::unordered_map<std::string, unsigned int> pattern_to_id;
    std::vector<State> states;
    std::vector<size_t> failure;
    size_t current_state;
    std::string buffer;
    size_t max_buffer_size;
    unsigned cur_id = 0;

    std::string to_lower(const std::string& s) {
        std::string result = s;
        std::transform(result.begin(), result.end(), result.begin(),
                      [](const unsigned char c){ return std::tolower(c); });
        return result;
    }

    unsigned add_matches(const std::set<std::string>& pattern_set) {
        for (const auto& pattern : pattern_set) {
            std::string processed_pattern = case_sensitive ? pattern : to_lower(pattern);
            patterns.insert(processed_pattern);
            pattern_to_id[processed_pattern] = cur_id;
        }

        size_t max_length = 0;
        for (const auto& pattern : patterns) {
            max_length = std::max(max_length, pattern.length());
        }
        max_buffer_size = max_length * 2;

        build_automaton();
        reset();

        return cur_id++;
    }

    void remove_matches(const unsigned int id) {
        std::vector<std::string> patterns_to_remove;
        for (const auto& [pattern, pattern_id] : pattern_to_id) {
            if (pattern_id == id) {
                patterns_to_remove.push_back(pattern);
            }
        }

        for (const auto& pattern : patterns_to_remove) {
            patterns.erase(pattern);
            pattern_to_id.erase(pattern);
        }

        if (!patterns.empty()) {
            size_t max_length = 0;
            for (const auto& pattern : patterns) {
                max_length = std::max(max_length, pattern.length());
            }
            max_buffer_size = max_length * 2;
        } else {
            max_buffer_size = 0;
        }

        build_automaton();
        reset();
    }

    void build_automaton() {
        states.clear();
        states.push_back(State());

        for (const auto& pattern : patterns) {
            size_t current = 0;

            for (unsigned char c : pattern) {
                if (states[current].goto_transitions.find(c) != states[current].goto_transitions.end()) {
                    current = states[current].goto_transitions[c];
                } else {
                    states.push_back(State());
                    const size_t new_state = states.size() - 1;
                    states[current].goto_transitions[c] = new_state;
                    current = new_state;
                }
            }

            states[current].output.insert(pattern);
        }

        failure.resize(states.size(), 0);
        std::queue<size_t> q;

        for (const auto& [c, s] : states[0].goto_transitions) {
            failure[s] = 0;
            q.push(s);
        }

        while (!q.empty()) {
            const size_t r = q.front();
            q.pop();

            for (const auto& [c, s] : states[r].goto_transitions) {
                q.push(s);

                size_t state = failure[r];
                while (state != 0 && states[state].goto_transitions.find(c) == states[state].goto_transitions.end()) {
                    state = failure[state];
                }

                if (states[state].goto_transitions.find(c) != states[state].goto_transitions.end()) {
                    failure[s] = states[state].goto_transitions[c];

                    for (const auto& output : states[failure[s]].output) {
                        states[s].output.insert(output);
                    }
                } else {
                    failure[s] = 0;
                }
            }
        }
    }

    std::set<unsigned int> process_token(const std::string& token) {
        std::string processed_token = token;
        if (!case_sensitive) {
            processed_token = to_lower(processed_token);
        }

        buffer += processed_token;

        if (buffer.length() > max_buffer_size) {
            const size_t excess = buffer.length() - max_buffer_size;
            buffer = buffer.substr(excess);
        }

        const size_t token_start_pos = buffer.length() - processed_token.length();
        size_t state = current_state;
        std::set<unsigned int> matched_ids;

        for (size_t i = 0; i < buffer.length(); ++i) {
            unsigned char c = buffer[i];

            if (states[state].goto_transitions.find(c) != states[state].goto_transitions.end()) {
                state = states[state].goto_transitions[c];
            } else {
                while (state != 0 && states[state].goto_transitions.find(c) == states[state].goto_transitions.end()) {
                    state = failure[state];
                }

                if (states[state].goto_transitions.find(c) != states[state].goto_transitions.end()) {
                    state = states[state].goto_transitions[c];
                } else {
                    state = 0;
                }
            }

            if (!states[state].output.empty()) {
                for (const auto& pattern : states[state].output) {
                    const size_t match_end = i;
                    if (match_end >= token_start_pos) {
                        matched_ids.insert(pattern_to_id.at(pattern));
                    }
                }
            }
        }

        current_state = state;
        return matched_ids;
    }

    void reset() {
        current_state = 0;
        buffer.clear();
    }
};
#endif // FORWARD_SEQUENCE_FILTER_HPP