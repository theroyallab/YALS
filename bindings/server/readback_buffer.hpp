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

    std::mutex readback_mutex;
    std::atomic<bool> being_destroyed {false};
};

// C API
bool readback_is_buffer_finished(ReadbackBuffer* buffer) {
    if (!buffer) return true;
    std::lock_guard lock(buffer->readback_mutex);
    const auto status = buffer->buffer_finished_write && buffer->last_readback_index >= buffer->ids->size();
    return status;
}

// C API
ReadbackBuffer* readback_create_buffer() {
    return new ReadbackBuffer{};
}

// C API
bool readback_read_next(ReadbackBuffer* buffer, char** outChar, llama_token* outToken) {
    if (!buffer || buffer->last_readback_index >= buffer->ids->size() || buffer->last_readback_index >= buffer->data->size()) {
        return false;
    }
    std::lock_guard lock(buffer->readback_mutex);
    *outChar = buffer->data->at(buffer->last_readback_index);
    *outToken = buffer->ids->at(buffer->last_readback_index);
    buffer->last_readback_index++;
    return true;
}

// C API
char* readback_read_status(ReadbackBuffer* buffer) {
    if (!buffer) return nullptr;
    std::lock_guard lock(buffer->readback_mutex);
    return buffer->status_buffer;
}

// C API
void readback_annihilate(ReadbackBuffer* buffer) {
    if (!buffer || !buffer->data || !buffer->ids)
        return;

    {
        std::lock_guard lock(buffer->readback_mutex);
        buffer->being_destroyed = true;
        for (char* str : *(buffer->data)) {
            free(str);  // memory was created via strdup which is a malloc.
        }
        free(buffer->status_buffer);
        delete buffer->data;
        delete buffer->ids;
    }

    delete buffer;
}

// Internal -- MALLOC copy -- Free all data buffers via free()
void readback_write_to_buffer(ReadbackBuffer* buffer, const std::string& data, const llama_token token) {
    if (!buffer || buffer->being_destroyed)
        return;

    std::lock_guard lock(buffer->readback_mutex);
    if (!buffer || buffer->being_destroyed)
        return;

    char* copy = strdup(data.c_str());
    buffer->data->push_back(copy);
    buffer->ids->push_back(token);
}

// Internal -- MALLOC copy -- Free status buffer via free()
void readback_finish(ReadbackBuffer* buffer, const std::string& status) {
    if (!buffer || buffer->being_destroyed)
        return;

    std::lock_guard lock(buffer->readback_mutex);
    if (!buffer || buffer->being_destroyed)
        return;

    char* copy = strdup(status.c_str());
    buffer->buffer_finished_write = true;
    buffer->status_buffer = copy;
}

#endif // READBACK_BUFFER_HPP