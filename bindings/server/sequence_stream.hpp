#ifndef SEQUENCE_STREAM_HPP
#define SEQUENCE_STREAM_HPP
#include <string>
#include <unordered_set>
#include <vector>
#include <optional>
#include "llama.h"
#include "trie.hpp"

/*
 *  The sequence stream is responsible for monitoring sequence events in the inference stream.
 *
 *  Provides:
 *  A lightweight buffer that indicates the status of the stream and how the processor should proceed.
 *
 *  Mechanism
 *  A sequence buffer and matching trie that checks for stops or rewinds, and indicates when we should buffer inputs.
 */

class SequenceStream {
    std::unordered_set<llama_token> stop_tokens;
    MatchTrie* match_trie = nullptr;

public:
    std::string sequence_buffer;

    enum class Continuation {
        ACCEPT,
        BUFFER,
        STOP,
        REWIND
    };

    SequenceStream() = default;

    void bind_sequences(const std::vector<std::string>& stop_seq, const std::vector<std::string>& rewind_seq,
             std::optional<std::vector<llama_token>> stop_tokens = std::nullopt) {
        // Delete nullptr is safe
        delete match_trie;
        match_trie = new MatchTrie();
        match_trie->AddMatchableWords(stop_seq, MatchType::STOP);
        match_trie->AddMatchableWords(rewind_seq, MatchType::REWIND);

        this->stop_tokens.clear();
        if (stop_tokens.has_value()) {
            this->stop_tokens = std::unordered_set(stop_tokens->begin(), stop_tokens->end());
        }

        this->sequence_buffer.clear();
    }

    Continuation append(const std::string_view& next_item, const llama_token last_token, std::string& out_sequence) {
        //Strip leading whitespace
        if (sequence_buffer.empty()) {
            sequence_buffer = next_item;
        } else {
            sequence_buffer += next_item;
        }

        if (stop_tokens.count(last_token) > 0) {
            return Continuation::STOP;
        }

        std::string stripped = sequence_buffer;
        if (!stripped.empty() && std::isspace(static_cast<unsigned char>(stripped.front()))) {
            stripped.erase(0, 1);
        }
        const auto result = match_trie->CheckBuffer(stripped);
        switch (result) {
            case MatchResult::MAYBE:
                return Continuation::BUFFER;

            case MatchResult::MATCHED_REWIND:
                out_sequence = std::move(sequence_buffer);
                sequence_buffer.clear();
                return Continuation::REWIND;

            case MatchResult::MATCHED_STOP:
                sequence_buffer.clear();
                return Continuation::STOP;

            case MatchResult::NO:
                out_sequence = std::move(sequence_buffer);
                sequence_buffer.clear();
                return Continuation::ACCEPT;
        }
        return Continuation::BUFFER;
    }
};

#endif // SEQUENCE_STREAM_HPP