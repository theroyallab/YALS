#ifndef SEQUENCE_STREAM_HPP
#define SEQUENCE_STREAM_HPP
#include <string>
#include <vector>
#include <unordered_set>
#include "trie.hpp"

class SequenceStream {
    int buffered_seq_size {};
    MatchTrie* match_trie = nullptr;

public:
    std::string sequence_buffer;

    enum SequenceStatus {
        ACCEPT = 1,
        BUFFER = 2,
        STOP = 4,
        REWIND = 8,
        RULE = 16
    };

    struct SequenceContext {
        SequenceStatus sequence_status {};
        int current_sequence_size {};
        std::string current_text_piece {};
        std::string current_sequence {};
        std::string unmatched_sequence {};
        std::unordered_set<int> matched_ids {};
    };

    SequenceStream() = default;

    ~SequenceStream() {
        delete match_trie;
    }

    void bind_sequences(const std::vector<std::string>& stop_seq, const std::vector<std::string>& rewind_seq) {
        delete match_trie;
        match_trie = new MatchTrie();

        if (!stop_seq.empty()) {
            match_trie->add_matchable_words(stop_seq, MATCH_ID_STOP);
        }
        if (!rewind_seq.empty()) {
            match_trie->add_matchable_words(rewind_seq, MATCH_ID_REWIND);
        }

        this->sequence_buffer.clear();
    }

    int bind_sequence(const std::vector<std::string>& sequences) {
        if (!match_trie) {
            match_trie = new MatchTrie();
        }
        return match_trie->add_matchable_words(sequences);
    }

    void remove_sequence(const int match_id) {
        if (match_trie) {
            match_trie->remove_matchable_words(match_id);
        }
    }

    SequenceContext append(const std::string_view& next_item) {
        sequence_buffer += next_item;
        buffered_seq_size++;

        const auto [result, matched_ids, unmatched, remainder] = match_trie->check_buffer(sequence_buffer);
        auto status = SequenceStatus::BUFFER;

        switch (result) {
            case MatchResult::NO:
                status = SequenceStatus::ACCEPT;
                break;
            case MatchResult::MAYBE:
                status = SequenceStatus::BUFFER;
                break;
            case MatchResult::MATCHED:
                if (matched_ids.count(MATCH_ID_STOP)) {
                    status = SequenceStatus::STOP;
                } else if (matched_ids.count(MATCH_ID_REWIND)) {
                    status = SequenceStatus::REWIND;
                } else {
                    status = SequenceStatus::RULE;
                }
                break;
        }

        const auto seq_ctx = SequenceContext{
            status,
            buffered_seq_size,
            std::string(next_item),
            std::string(sequence_buffer),
            std::string(unmatched),
            matched_ids};

        if (result != MatchResult::MAYBE) {
            buffered_seq_size = 0;
            if (result == MatchResult::MATCHED) {
                sequence_buffer = std::string(remainder);
            } else {
                sequence_buffer.clear();
            }
        }

        return seq_ctx;
    }

    bool has_trie() const {
        return match_trie != nullptr;
    }

    void reset() {
        delete match_trie;
        match_trie = nullptr;
        sequence_buffer.clear();
        buffered_seq_size = 0;
    }
};

#endif // SEQUENCE_STREAM_HPP
