#ifndef MATCH_TRIE_HPP
#define MATCH_TRIE_HPP

#include <string_view>
#include <unordered_map>
#include <memory>
#include <cctype>

namespace MatchTrie {
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

        TrieNode() : isEndOfWord(false) {}
    };

    class MatchTrie {
        std::unique_ptr<TrieNode> root;

        // Helper function to convert char to lowercase
        static char ToLower(char c) {
            return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }

    public:
        MatchTrie() : root(std::make_unique<TrieNode>()) {}

        void AddMatchableWords(const char** words, const size_t count, const MatchType type) const {
            for (size_t i = 0; i < count; ++i) {
                TrieNode* current = root.get();

                for (const char* c = words[i]; *c; ++c) {
                    char lowerChar = ToLower(*c);
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
            TrieNode* current = root.get();

            for (char c : buffer) {
                char lowerChar = ToLower(c);
                if (current->children.find(lowerChar) == current->children.end()) {
                    return MatchResult::NO;
                }
                current = current->children[lowerChar].get();
            }

            if (current->isEndOfWord) {
                return (current->matchType == MatchType::REWIND) ?
                       MatchResult::MATCHED_REWIND :
                       MatchResult::MATCHED_STOP;
            }

            return MatchResult::MAYBE;
        }
    };
}


#endif // MATCH_TRIE_HPP