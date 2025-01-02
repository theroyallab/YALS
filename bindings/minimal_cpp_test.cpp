#include "binding.cpp"
#include <string>
#include <thread>
#include <chrono>

int main() {
    const std::string modelPath = "/home/blackroot/Desktop/YALS/bindings/gguf/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf";
    const auto modelLayers = 999;
    const auto ctxLen = 8192;
    const auto prompt = "This is the test prompt";
    const auto numTokens = 35;

    const auto model = LoadModel(modelPath.c_str(), modelLayers, nullptr);
    const auto ctx = InitiateCtx(model, ctxLen, 1, true, 0, 0);
    const auto sampler = MakeSampler();
    GreedySampler(sampler);

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
        const char* next = ReadbackNext(readbackBuffer);
        if (next != nullptr) {
            std::cout << next;
            std::flush(std::cout);
        }
    }

    FreeModel(model);
    FreeSampler(sampler);
    ResetReadbackBuffer(readbackBuffer);
}