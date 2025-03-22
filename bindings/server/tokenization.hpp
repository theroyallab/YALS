#ifndef TOKENIZATION_HPP
#define TOKENIZATION_HPP

#include <string>
#include <vector>
#include <optional>
#include <string_view>
#include "llama.h"
#include "common.h"

// From Llama cpp server example
static size_t validate_utf8(const std::string& text) {
    const size_t len = text.size();
    if (len == 0) return 0;

    for (size_t i = 1; i <= 4 && i <= len; ++i) {
        const unsigned char c = text[len - i];
        if ((c & 0xE0) == 0xC0) {
            // 110xxxxx
            if (i < 2) return len - i;
        } else if ((c & 0xF0) == 0xE0) {
            // 1110xxxx
            if (i < 3) return len - i;
        } else if ((c & 0xF8) == 0xF0) {
            // 11110xxx
            if (i < 4) return len - i;
        }
    }

    return len;
}

class TokenStreamDetokenizer {
    std::string utf_buffer;
    llama_context* ctx;

public:
    explicit TokenStreamDetokenizer(llama_context* ctx)
        : ctx(ctx) {
    }

    std::optional<std::string> process_token(const llama_token token, const bool parse_special) {
        const std::string piece = common_token_to_piece(ctx, token, parse_special);
        utf_buffer += piece;

        const size_t valid_bytes = validate_utf8(utf_buffer);

        if (valid_bytes == 0) {
            return std::string{};
        }

        if (valid_bytes == utf_buffer.size()) {
            std::string result = std::move(utf_buffer);
            utf_buffer.clear();
            return result;
        }

        std::string result = utf_buffer.substr(0, valid_bytes);
        utf_buffer = utf_buffer.substr(valid_bytes);
        return result;
    }

    std::string flush() {
        std::string result = std::move(utf_buffer);
        utf_buffer.clear();
        return result;
    }

    [[nodiscard]] bool has_incomplete() const {
        return !utf_buffer.empty();
    }

    void reset() {
        utf_buffer.clear();
    }
};

class Tokenizer {
    llama_context* ctx;
    const llama_vocab* vocab;

public:
    Tokenizer(const llama_model* model, llama_context* ctx)
        : ctx(ctx), vocab(llama_model_get_vocab(model)) {
    }

    [[nodiscard]] bool is_eos_token(const llama_token token) const {
        return token == llama_vocab_eos(vocab);
    }

    [[nodiscard]]std::vector<llama_token> tokenize(const std::string_view& text, const bool add_special = true, const bool parse_special = true) const {
        return common_tokenize(vocab, std::string(text), add_special, parse_special);
    }
};

#endif // TOKENIZATION_HPP