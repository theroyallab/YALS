#include "binding.cpp"
#include <string>
#include <thread>
#include <chrono>

int main() {
    const std::string modelPath = "D:\\koboldcpp\\allura-org_Bigger-Body-8b-Q6_K_L.gguf";
    const auto modelLayers = 999;
    const auto ctxLen = 8192;
    const auto prompt = "This is the test prompt";
    const auto numTokens = 200;

    const auto model = LoadModel(modelPath.c_str(), modelLayers, nullptr, nullptr);

    const auto ctx = InitiateCtx(
        model,
        ctxLen,
        modelLayers,
        512,
        true,
        0,
        false,
        1,
        1,
        -1);

    auto eotId = 0;
    std::cout << eotId << std::endl;
    auto eotPiece = llama_vocab_get_text(&model->vocab, eotId);
    std::cout << eotPiece << std::endl;

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