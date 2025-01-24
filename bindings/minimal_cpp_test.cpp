#include "binding.cpp"
#include <string>
#include <thread>
#include <chrono>

int main() {
    const std::string modelPath = "/home/blackroot/Desktop/YALS/bindings/gguf/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf";
    const auto modelLayers = 999;
    const auto ctxLen = 8192;
    const auto prompt = "This is the test prompt";
    const auto numTokens = 200;

    const auto model = LoadModel(modelPath.c_str(), modelLayers, nullptr, nullptr);

    const auto ctx = InitiateCtx(
        model,
        ctxLen,
        512,
        true,
        false,
        false,
        0,
        0,
        false,
        -1,
        -1,
        0,
        -1,
        -1,
        1,
        1,
        -1);

    const auto sampler = MakeSampler();
    GreedySampler(sampler);

    // int32_t* tokenized = EndpointTokenize(model, "this is a thing", false, false);
    //
    // // Get the length from first element
    // const int32_t length = tokenized[0];
    //
    // // Iterate over the actual tokens (skip the length element)
    // for (int i = 1; i <= length; i++) {
    //     std::cout << "Token: " << tokenized[i] << std::endl;
    // }
    //
    // const char* result = EndpointDetokenize(model, tokenized + 1, tokenized[0], 100, false, false);

    // std::cout << result << std::endl;
    // EndpointFreeTokens(tokenized);
    // EndpointFreeString(result);

    auto readbackBuffer = CreateReadbackBuffer();
    std::thread inferenceThread(
        [&]() {
            InferToReadbackBuffer(
                model,
                sampler,
                ctx,
                readbackBuffer,
                prompt,
                numTokens,
                true,
                true,
                nullptr,
                nullptr,
                0,
                nullptr,
                0
            );
        }
    );

    // Detach the thread if you don't need to wait for it
    inferenceThread.detach();

    while (!IsReadbackBufferDone(readbackBuffer)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        char* outChar;
        llama_token outTok;
        if (ReadbackNext(readbackBuffer, &outChar, &outTok)) {
            std::cout << outChar;
            std::cout << outTok;
            std::flush(std::cout);
        }
    }

    std::cout << readbackBuffer->jsonOutputBuffer;

    FreeModel(model);
    FreeSampler(sampler);
    ResetReadbackBuffer(readbackBuffer);
}