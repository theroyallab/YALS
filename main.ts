import { createApi } from "./api/server.ts";
import { setupLogger } from "./common/logging.ts";
import { loadModel } from "./common/model.ts";

if (import.meta.main) {
    await setupLogger();
    await loadModel("D:/koboldcpp/Meta-Llama-3-8B-Instruct-Q5_K_M.gguf", 999);
    createApi();
}
