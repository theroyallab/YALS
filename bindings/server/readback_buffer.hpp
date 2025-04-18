#ifndef READBACK_BUFFER_HPP
#define READBACK_BUFFER_HPP

#include <vector>
#include <llama.h>
#include <cstring>
#include <mutex>

/**
 * Owned buffer for live token and character streaming.
 */
struct ReadbackBuffer {
    unsigned last_readback_index {0};
    bool buffer_finished_write {false};
    char* status_buffer = nullptr;

    // Owner internal char*'s. Must free all of them. (strdup)
    std::vector<char*>* data = new std::vector<char*>();
    std::vector<llama_token>* ids = new std::vector<llama_token>();

    // Two phase destruction
    std::mutex readback_mutex;
    std::atomic<bool> being_destroyed {false};
};

template<typename Callback>
bool using_readback_buffer(ReadbackBuffer* buffer, Callback&& callback) {
    if (!buffer || buffer->being_destroyed)
        return false;

    std::lock_guard lock(buffer->readback_mutex);
    if (!buffer || buffer->being_destroyed)
        return false;

    callback();
    return true;
}

// C API
bool readback_is_buffer_finished(ReadbackBuffer* buffer) {
    bool is_finished = true;
    using_readback_buffer(buffer, [&] {
        is_finished = buffer->buffer_finished_write && buffer->last_readback_index >= buffer->ids->size();
    });
    return is_finished;
}

// C API
ReadbackBuffer* readback_create_buffer() {
    return new ReadbackBuffer{};
}

// C API
bool readback_read_next(ReadbackBuffer* buffer, char** outChar, llama_token* outToken) {
    bool success = false;
    using_readback_buffer(buffer, [&] {
        if (buffer->last_readback_index < buffer->ids->size() &&
            buffer->last_readback_index < buffer->data->size()) {
            *outChar = buffer->data->at(buffer->last_readback_index);
            *outToken = buffer->ids->at(buffer->last_readback_index);
            buffer->last_readback_index++;
            success = true;
        }
    });

    return success;
}

// C API
char* readback_read_status(ReadbackBuffer* buffer) {
    char* status = nullptr;
    using_readback_buffer(buffer, [&]() {
        status = buffer->status_buffer;
    });

    return status;
}

// C API
void readback_annihilate(ReadbackBuffer* buffer) {
    if (!buffer)
        return;

    {
        std::lock_guard lock(buffer->readback_mutex);
        buffer->being_destroyed = true;

        if (buffer->data) {
            for (char* str : *(buffer->data)) {
                free(str);
            }
            delete buffer->data;
        }

        if (buffer->ids) {
            delete buffer->ids;
        }

        if (buffer->status_buffer) {
            free(buffer->status_buffer);
        }
    }
    delete buffer;
}

// Internal -- MALLOC copy -- Free all data buffers via free()
void readback_write_to_buffer(ReadbackBuffer* buffer, const std::string& data, const llama_token token) {
    using_readback_buffer(buffer, [&]() {
        char* copy = strdup(data.c_str());
        buffer->data->push_back(copy);
        buffer->ids->push_back(token);
    });
}

// Internal -- MALLOC copy -- Free status buffer via free()
void readback_finish(ReadbackBuffer* buffer, const std::string& status) {
    using_readback_buffer(buffer, [&]() {
        char* copy = strdup(status.c_str());
        if (buffer->status_buffer) {
            free(buffer->status_buffer);
        }
        buffer->buffer_finished_write = true;
        buffer->status_buffer = copy;
    });
}

#endif // READBACK_BUFFER_HPP