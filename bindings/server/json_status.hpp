#ifndef JSON_STATUS_HPP
#define JSON_STATUS_HPP

#include <iomanip>
#include <sstream>
#include "slot.hpp"

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
                    ss << "\\u" << std::setw(4) << static_cast<int>(ch);
                } else {
                    ss << ch;
                }
        }
    }

    return ss.str();
}

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

inline std::string make_empty_json_status_string(const std::string &finish_reason,
                                          const std::string &stop_token) {
    constexpr double prompt_sec = 0.0;
    constexpr double gen_sec = 0.0;
    constexpr double total_sec = 0.0;
    constexpr double prompt_tokens_per_sec = 0.0;
    constexpr double gen_tokens_per_sec = 0.0;
    constexpr int prompt_tokens = 0;
    constexpr int gen_tokens = 0;
    constexpr int slot_id = -1;
    constexpr int request_id = -1;
    constexpr int job_index = -1;

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6) << "{";

    add_json_value(ss, "slotId", slot_id);
    add_json_value(ss, "slotRequestId", request_id);
    add_json_value(ss, "jobIndex", job_index);

    add_json_value(ss, "promptTokens", prompt_tokens);
    add_json_value(ss, "genTokens", gen_tokens);

    add_json_value(ss, "promptSec", prompt_sec);
    add_json_value(ss, "genSec", gen_sec);
    add_json_value(ss, "totalSec", total_sec);
    add_json_value(ss, "genTokensPerSec", gen_tokens_per_sec);
    add_json_value(ss, "promptTokensPerSec", prompt_tokens_per_sec);

    add_json_value(ss, "finishReason", finish_reason);
    add_json_value(ss, "stopToken", stop_token, true);

    ss << "}";

    return ss.str();
}

inline std::string make_json_status_string(const Slot& slot, const std::string &finish_reason,
                                          const std::string &stop_token) {

    const double prompt_sec = (slot.prompt_end_time - slot.slot_start_time) / 1000.0;
    const double gen_sec = (slot.generating_end_time - slot.prompt_end_time) / 1000.0;
    const double total_sec = (slot.generating_end_time - slot.slot_start_time) / 1000.0;


    const double prompt_tokens_per_sec = prompt_sec > 0 ?
        static_cast<double>(slot.prompt_tokens_processed) / prompt_sec : 0.0;
    const double gen_tokens_per_sec = gen_sec > 0 ?
        static_cast<double>(slot.tokens_generated) / gen_sec : 0.0;

    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2) << "{";

    add_json_value(ss, "slotId", slot.slot_id);
    add_json_value(ss, "slotRequestId", slot.request_id);
    add_json_value(ss, "jobIndex", slot.job_index);

    add_json_value(ss, "promptTokens", slot.prompt_tokens_processed);
    add_json_value(ss, "genTokens", slot.tokens_generated);

    add_json_value(ss, "promptSec", prompt_sec);
    add_json_value(ss, "genSec", gen_sec);
    add_json_value(ss, "totalSec", total_sec);
    add_json_value(ss, "genTokensPerSec", gen_tokens_per_sec);
    add_json_value(ss, "promptTokensPerSec", prompt_tokens_per_sec);

    add_json_value(ss, "finishReason", finish_reason);
    add_json_value(ss, "stopToken", stop_token, true);

    ss << "}";

    return ss.str();
}

#endif // JSON_STATUS_HPP