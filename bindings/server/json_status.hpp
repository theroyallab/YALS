#ifndef JSON_HPP
#define JSON_HPP
#include <iomanip>
#include <sstream>
#include "llama.h"

inline std::string escape_string(const std::string& input) {
    std::ostringstream ss;
    ss << std::hex << std::setfill('0');

    for (const unsigned char ch : input) {
        switch (ch) {
            case '"':  ss << "\\\""; break;
            case '\\': ss << "\\\\"; break;
            case '\b': ss << "\\b";  break;
            case '\f': ss << "\\f";  break;
            case '\n': ss << "\\n";  break;
            case '\r': ss << "\\r";  break;
            case '\t': ss << "\\t";  break;
            default:
                if (ch < 0x20) {
                    // Use consistent formatting for control characters
                    ss << "\\u" << std::setw(4) << static_cast<int>(ch);
                } else {
                    ss << ch;
                }
        }
    }

    return ss.str();
}

// Helper function to build JSON string entries safely
template<typename T>
void add_json_value(std::ostringstream& ss, const std::string& key, const T& value, bool is_last = false) {
    ss << "\"" << key << "\":";
    if constexpr (std::is_same_v<T, std::string>) {
        ss << "\"" << escape_string(value) << "\"";
    } else {
        ss << value;
    }

    if (!is_last) {
        ss << ",";
    }
}

inline std::string make_json_status_string(const llama_context* context, const std::string &finish_reason,
                                           const std::string &stop_token) {
    const auto [t_start_ms, t_load_ms, t_p_eval_ms, t_eval_ms, n_p_eval, n_eval] = llama_perf_context(context);

    const double prompt_sec = t_p_eval_ms / 1000.0;
    const double gen_sec = t_eval_ms / 1000.0;

    const double prompt_tokens_per_sec = prompt_sec > 0 ? n_p_eval / prompt_sec : 0;
    const double gen_tokens_per_sec = gen_sec > 0 ? n_eval / gen_sec : 0;

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6) << "{";

    add_json_value(ss, "promptTokens", n_p_eval);
    add_json_value(ss, "genTokens", n_eval);
    add_json_value(ss, "promptSec", prompt_sec);
    add_json_value(ss, "genSec", gen_sec);
    add_json_value(ss, "genTokensPerSec", gen_tokens_per_sec);
    add_json_value(ss, "promptTokensPerSec", prompt_tokens_per_sec);
    add_json_value(ss, "finishReason", finish_reason);
    add_json_value(ss, "stopToken", stop_token, true);

    ss << "}";

    return ss.str();
}

#endif //JSON_HPP
