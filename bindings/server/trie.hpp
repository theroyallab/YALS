#ifndef MATCH_TRIE_HPP
#define MATCH_TRIE_HPP

#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <cctype>
#include <vector>
#include <iostream>

enum class MatchResult {
    NO,
    MAYBE,
    MATCHED
};

constexpr int MATCH_ID_STOP = 1;
constexpr int MATCH_ID_REWIND = 2;

class TrieNode {
public:
    std::unordered_map<char, std::unique_ptr<TrieNode>> children;
    std::unordered_set<int> match_ids;

    TrieNode() = default;

    bool is_end_of_word() const {
        return !match_ids.empty();
    }
};

class MatchTrie {
    std::unique_ptr<TrieNode> root;
    std::unordered_map<int, std::vector<std::string>> id_to_sequences;
    int next_auto_id = 3;

    static char to_lower(const char c) {
        return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }

    void remove_sequence_from_trie(const std::string& word, const int match_id) {
        TrieNode* current = root.get();
        std::vector<TrieNode*> path;
        std::vector<char> chars;

        for (const char c : word) {
            char lower_char = to_lower(c);
            auto it = current->children.find(lower_char);
            if (it == current->children.end()) {
                return;
            }
            path.push_back(current);
            chars.push_back(lower_char);
            current = it->second.get();
        }

        current->match_ids.erase(match_id);

        if (current->match_ids.empty() && current->children.empty()) {
            for (int i = static_cast<int>(path.size()) - 1; i >= 0; --i) {
                TrieNode* parent = path[i];
                char ch = chars[i];
                TrieNode* child = (i == static_cast<int>(path.size()) - 1) ? current : path[i + 1];

                if (child->match_ids.empty() && child->children.empty()) {
                    parent->children.erase(ch);
                } else {
                    break;
                }
            }
        }
    }

public:
    MatchTrie() : root(std::make_unique<TrieNode>()) {}

    int add_matchable_words(const std::vector<std::string>& words) {
        const int match_id = next_auto_id++;
        add_matchable_words(words, match_id);
        return match_id;
    }

    void add_matchable_words(const std::vector<std::string>& words, const int match_id) {
        for (const auto& word : words) {
            TrieNode* current = root.get();

            for (const char c : word) {
                char lower_char = to_lower(c);
                if (current->children.find(lower_char) == current->children.end()) {
                    current->children[lower_char] = std::make_unique<TrieNode>();
                }
                current = current->children[lower_char].get();
            }
            current->match_ids.insert(match_id);
        }

        if (id_to_sequences.find(match_id) == id_to_sequences.end()) {
            id_to_sequences[match_id] = words;
        } else {
            id_to_sequences[match_id].insert(
                id_to_sequences[match_id].end(),
                words.begin(),
                words.end()
            );
        }
    }

    void remove_matchable_words(const int match_id) {
        auto it = id_to_sequences.find(match_id);
        if (it == id_to_sequences.end()) {
            return;
        }

        for (const auto& word : it->second) {
            remove_sequence_from_trie(word, match_id);
        }

        id_to_sequences.erase(it);
    }

    struct BufferCheckResult {
        MatchResult result;
        std::unordered_set<int> matched_ids;
        std::string_view unmatched;
        std::string_view remainder;
    };

    //TODO:: @Z Potential optimization is to store unmatchable parts of the buffer if similar to priors, which would greatly decrease
    //the expense of this op.
    [[nodiscard]] BufferCheckResult check_buffer(const std::string_view& buffer) const {
        std::unordered_set<int> best_matched_ids;
        std::string_view content_before_match = buffer;
        std::string_view remainder_after_match = "";
        size_t earliest_match_start = buffer.length();
        bool has_partial_match = false;

        for (size_t start = 0; start < buffer.length(); ++start) {
            TrieNode* node = root.get();
            size_t pos = start;

            for (; pos < buffer.length(); ++pos) {
                char lower_char = to_lower(buffer[pos]);

                auto it = node->children.find(lower_char);
                if (it == node->children.end()) {
                    break;
                }

                node = it->second.get();

                if (node->is_end_of_word()) {
                    if (start < earliest_match_start) {
                        earliest_match_start = start;
                        best_matched_ids = node->match_ids;
                        content_before_match = buffer.substr(0, start);
                        remainder_after_match = buffer.substr(pos + 1);
                    } else if (start == earliest_match_start) {
                        best_matched_ids.insert(node->match_ids.begin(), node->match_ids.end());
                    }
                }
            }

            if (pos == buffer.length() && !node->children.empty()) {
                has_partial_match = true;
            }
        }

        if (!best_matched_ids.empty()) {
            return {MatchResult::MATCHED, best_matched_ids, content_before_match, remainder_after_match};
        } else if (has_partial_match) {
            return {MatchResult::MAYBE, {}, buffer, ""};
        } else {
            return {MatchResult::NO, {}, buffer, ""};
        }
    }
};

#endif // MATCH_TRIE_HPP
