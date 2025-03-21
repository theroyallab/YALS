#ifndef READBACK_BUFFER_HPP
#define READBACK_BUFFER_HPP

#include <vector>
#include <llama.h>
#include <cstring>

/**
 * ReadbackBuffer is a data structure for buffering and retrieving
 * generated tokens and their text representations.
 */
struct ReadbackBuffer {
    unsigned last_readback_index {0};
    bool buffer_finished_write {false};
    char* status_buffer = nullptr;
    std::vector<char*>* data = new std::vector<char*>();
    std::vector<llama_token>* ids = new std::vector<llama_token>();
};

// C API
bool readback_is_buffer_finished(const ReadbackBuffer* buffer) {
    return buffer->buffer_finished_write;
}

// C API
ReadbackBuffer* readback_create_buffer() {
    return new ReadbackBuffer{};
}

// C API
bool readback_read_next(ReadbackBuffer* buffer, char** outChar, llama_token* outToken) {
    if (buffer->last_readback_index >= buffer->data->size()) {
        return false;
    }

    *outChar = buffer->data->at(buffer->last_readback_index);
    *outToken = buffer->ids->at(buffer->last_readback_index);
    buffer->last_readback_index++;
    return true;
}

// C API
char* readback_read_status(const ReadbackBuffer* buffer) {
    return buffer->status_buffer;
}

// C API
void readback_reset(ReadbackBuffer* buffer) {
    // Ensure we entirely free the string data we allocated. We own this data.
    for (char* str : *(buffer->data)) {
        free(str);  // memory was created via strdup which is a malloc.
    }

    buffer->data->clear();
    buffer->ids->clear();

    if (buffer->status_buffer != nullptr) {
        free(buffer->status_buffer);
        buffer->status_buffer = nullptr;
    }

    buffer->last_readback_index = 0;
    buffer->buffer_finished_write = false;
}

// C API
void readback_annihilate(ReadbackBuffer* buffer) {
    readback_reset(buffer);

    delete buffer->data;
    delete buffer->ids;

    delete buffer;
}

// Internal -- MALLOC copy -- Free all data buffers via free()
void readback_write_to_buffer(const ReadbackBuffer* buffer, const std::string& data, const llama_token token) {
    char* copy = strdup(data.c_str());

    buffer->data->push_back(copy);
    buffer->ids->push_back(token);
}

// Internal -- MALLOC copy -- Free status buffer via free()
void readback_finish(ReadbackBuffer* buffer, const std::string& status) {
    char* copy = strdup(status.c_str());

    buffer->buffer_finished_write = true;
    buffer->status_buffer = copy;
}

#endif // READBACK_BUFFER_HPP