#ifndef REQUEST_HPP
#define REQUEST_HPP

/*
 * A light abstraction over a request to fill a slot. This pends in a queue until we have free slots to take
 * the next request.
 */

struct Request {
    int id;
    std::vector<llama_token> prompt_tokens;
    InferenceArgs inference_args;
};

#endif // REQUEST_HPP