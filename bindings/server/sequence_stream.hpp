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
    int buffered_seq_size {};
    MatchTrie* match_trie = nullptr;

public:
    std::string sequence_buffer;

    enum SequenceStatus {
        ACCEPT = 1,
        BUFFER = 2,
        STOP = 4,
        REWIND = 8
    };

    // Contains the result of what was in the buffer during the status.
    struct SequenceContext {
        SequenceStatus sequence_status {};
        int current_sequence_size {};
        std::string current_text_piece {};
        std::string current_sequence {};
        std::string unmatched_sequence {};
    };

    SequenceStream() = default;

    void bind_sequences(const std::vector<std::string>& stop_seq, const std::vector<std::string>& rewind_seq) {
        // Delete nullptr is safe
        delete match_trie;
        match_trie = new MatchTrie();
        match_trie->add_matchable_words(stop_seq, MatchType::STOP);
        match_trie->add_matchable_words(rewind_seq, MatchType::REWIND);

        this->sequence_buffer.clear();
    }

    SequenceContext append(const std::string_view& next_item) {
        sequence_buffer += next_item;
        buffered_seq_size++;

        const auto [result, unmatched] = match_trie->check_buffer(sequence_buffer);
        auto status = SequenceStatus::BUFFER;
        switch (result) {
            case MatchResult::NO:
                status = SequenceStatus::ACCEPT;
                break;
            case MatchResult::MAYBE:
                status = SequenceStatus::BUFFER;
                break;
            case MatchResult::MATCHED_REWIND:
                status = SequenceStatus::REWIND;
                break;
            case MatchResult::MATCHED_STOP:
                status = SequenceStatus::STOP;
                break;
        }

        const auto seq_ctx = SequenceContext{
            status,
            buffered_seq_size,
            std::string(next_item),
            std::string(sequence_buffer),
            std::string(unmatched)};

        if (result != MatchResult::MAYBE) {
            buffered_seq_size = 0;
            sequence_buffer.clear();
        }

        return seq_ctx;
    }
};

#endif // SEQUENCE_STREAM_HPP