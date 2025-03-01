#ifndef MATCH_TRIE_HPP
#define MATCH_TRIE_HPP

#include <string_view>
#include <unordered_map>
#include <memory>
#include <cctype>
#include <iostream>

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

    struct MatchInfo {
        MatchResult result;
        size_t matchPos;
        size_t matchLength;

        MatchInfo(MatchResult r = MatchResult::NO, size_t pos = std::string::npos, size_t len = 0):
            result(r), matchPos(pos), matchLength(len) {}
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
        std::string cachedPrefix;

        // Helper function to convert char to lowercase
        static char ToLower(char c) {
            return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }

        // Helper method to print all words in trie
        void PrintWordsHelper(TrieNode* node, std::string prefix) const {
            if (node->isEndOfWord) {
                std::cout << "  - '" << prefix << "' (";
                std::cout << (node->matchType == MatchType::STOP ? "STOP" : "REWIND");
                std::cout << ")" << std::endl;
            }
            
            for (const auto& pair : node->children) {
                PrintWordsHelper(pair.second.get(), prefix + pair.first);
            }
        }

        void updateCachedPrefix(MatchInfo info, const std::string_view& buffer) {
            if (info.matchPos != std::string::npos) {
                cachedPrefix = buffer.substr(0, info.matchPos);
            } else {
                cachedPrefix.clear();
            }
        }

    public:
        MatchTrie() : root(std::make_unique<TrieNode>()) {}

        // Add this method to print all words in the trie
        void PrintWords() const {
            std::cout << "Words in trie:" << std::endl;
            PrintWordsHelper(root.get(), "");
            std::cout << "-------------------" << std::endl;
        }

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

        [[nodiscard]] MatchInfo CheckBuffer(const std::string_view& buffer) {
            MatchInfo info;
            const auto initialPos = buffer.rfind(cachedPrefix, 0) == 0 ? cachedPrefix.length() : 0;

            // Look for substring matches
            for (size_t startPos = initialPos; startPos < buffer.length(); ++startPos) {
                TrieNode* current = root.get();
                bool potentialMatch = false;
                size_t matchLength = 0;

                for (size_t i = startPos; i < buffer.length(); ++i) {
                    char lowerChar = ToLower(buffer[i]);
                    auto it = current->children.find(lowerChar);

                    // Move to the next position
                    if (it == current->children.end()) {
                        break;
                    }

                    current = it->second.get();
                    potentialMatch = true;
                    matchLength++;

                    // Complete match of stop/rewind
                    if (current->isEndOfWord) {
                        const MatchResult endResult = (current->matchType == MatchType::REWIND) ?
                               MatchResult::MATCHED_REWIND :
                               MatchResult::MATCHED_STOP;

                        info = MatchInfo(endResult, startPos, matchLength);
                        cachedPrefix = buffer;
                        return info;
                    }
                }

                // If there is a possible match, but there's no end of word, mark as maybe
                if (potentialMatch && info.result == MatchResult::NO) {
                    info = MatchInfo(MatchResult::MAYBE, startPos, matchLength);
                }
            }

            cachedPrefix = buffer;
            return info;
        }
    };
}


#endif // MATCH_TRIE_HPP