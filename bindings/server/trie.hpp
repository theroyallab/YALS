#ifndef MATCH_TRIE_HPP
#define MATCH_TRIE_HPP

#include <string_view>
#include <unordered_map>
#include <memory>
#include <cctype>

/*
 * A simple prefix tree for efficient sequence to sequence matches.
 */

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
    bool isEndOfWord;
    MatchType matchType;

    TrieNode() : isEndOfWord(false), matchType() {
    }
};

class MatchTrie {
    std::unique_ptr<TrieNode> root;

    static char ToLower(const char c) {
        return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }

public:
    MatchTrie() : root(std::make_unique<TrieNode>()) {}

    void AddMatchableWords(const std::vector<std::string>& words, const MatchType type) const {
        for (const auto& word : words) {
            TrieNode* current = root.get();

            for (const char c : word) {
                char lowerChar = ToLower(c);
                if (current->children.find(lowerChar) == current->children.end()) {
                    current->children[lowerChar] = std::make_unique<TrieNode>();
                }
                current = current->children[lowerChar].get();
            }
            current->isEndOfWord = true;
            current->matchType = type;
        }
    }

    [[nodiscard]] MatchResult CheckBuffer(const std::string_view& buffer) const {
        TrieNode* node = root.get();

        // If the root node is blank, we can't possibly have any matches.
        if (node->children.empty())
            return MatchResult::NO;

        // Check each character in the buffer
        for (size_t i = 0; i < buffer.length(); ++i) {
            char lowerChar = ToLower(buffer[i]);

            // Look for this character in current node's children
            auto it = node->children.find(lowerChar);
            if (it == node->children.end()) {
                // Character not found, no match possible
                return MatchResult::NO;
            }

            // Move to the next node
            node = it->second.get();

            if (node->isEndOfWord) {
                return (node->matchType == MatchType::REWIND) ?
                    MatchResult::MATCHED_REWIND :
                    MatchResult::MATCHED_STOP;
            }
        }

        // If we've processed the entire buffer and haven't found a match yet,
        // but the current node has children, then there's a potential match
        if (!node->children.empty()) {
            return MatchResult::MAYBE;
        }

        return MatchResult::NO;
    }
};



#endif // MATCH_TRIE_HPP