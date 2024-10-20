#include <iomanip>
#include <iostream>

void PrintPerformanceInfo(const llama_context* context) {
    const auto data = llama_perf_context(context);

    float prompt_tok_per_sec = 1e3 / data.t_p_eval_ms * data.n_p_eval;
    float gen_tok_per_sec = 1e3 / data.t_eval_ms * data.n_eval;

    std::cout << "\n\n" << std::fixed << std::setprecision(2)
              << "Prompt Processing: " << prompt_tok_per_sec << " tok/s, "
              << "Text Generation: " << gen_tok_per_sec << " tok/s" << "\n" << std::endl;
}