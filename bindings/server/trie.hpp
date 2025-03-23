#ifndef MATCH_TRIE_HPP
#define MATCH_TRIE_HPP

#include <string_view>
#include <unordered_map>
#include <memory>
#include <cctype>

enum class MatchType {
    REWIND,
    STOP
};

enum class MatchResult {
    NO,
    MAYBE,
    MATCHED_REWIND,
    MATCHED_STOP
};

class TrieNode {
public:
    std::unordered_map<char, std::unique_ptr<TrieNode>> children;
    bool is_end_of_word;
    MatchType match_type;

    TrieNode() : is_end_of_word(false), match_type() {
    }
};

class MatchTrie {
    std::unique_ptr<TrieNode> root;

    static char to_lower(const char c) {
        return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }

public:
    MatchTrie() : root(std::make_unique<TrieNode>()) {}

    void add_matchable_words(const std::vector<std::string>& words, const MatchType type) const {
        for (const auto& word : words) {
            TrieNode* current = root.get();

            for (const char c : word) {
                char lower_char = to_lower(c);
                if (current->children.find(lower_char) == current->children.end()) {
                    current->children[lower_char] = std::make_unique<TrieNode>();
                }
                current = current->children[lower_char].get();
            }
            current->is_end_of_word = true;
            current->match_type = type;
        }
    }

    // Does substring matches to check submatches in the buffer, which is actually needed.
    [[nodiscard]] MatchResult check_buffer(const std::string_view& buffer) const {
        if (root->children.empty())
            return MatchResult::NO;

        MatchResult result = MatchResult::NO;

        for (size_t start = 0; start < buffer.length(); ++start) {
            TrieNode* node = root.get();
            size_t i = start;
            
            for (; i < buffer.length(); ++i) {
                char lower_char = to_lower(buffer[i]);
                
                auto it = node->children.find(lower_char);
                if (it == node->children.end()) {
                    break;
                }
                
                node = it->second.get();
                
                if (node->is_end_of_word) {
                    return (node->match_type == MatchType::REWIND) ?
                        MatchResult::MATCHED_REWIND :
                        MatchResult::MATCHED_STOP;
                }
            }
            
            if (i == buffer.length() && !node->children.empty() && result == MatchResult::NO) {
                result = MatchResult::MAYBE;
            }
        }
        
        return result;
    }
};

#endif // MATCH_TRIE_HPP